import numpy as np
from dataclasses import dataclass, field


@dataclass
class KFConfig:
    # IMU accelerometer noise std (m/s^2)
    imu_accel_noise_std: float = 0.1
    # IMU gyroscope noise std (rad/s)
    imu_gyro_noise_std: float = 0.01
    # Accelerometer bias random-walk std (m/s^2 per sqrt(s))
    accel_bias_rw_std: float = 0.001
    # Gyroscope bias random-walk std (rad/s per sqrt(s))
    gyro_bias_rw_std: float = 0.0001

    # VIO obs noise
    vio_pos_std: float = 0.20
    vio_vel_std: float = 0.30

    # PnP obs noise (more reliable)
    pnp_pos_std: float = 0.03

    # VIO delay compensation
    vio_delay_compensation: bool = False
    vio_delay_sec: float = 0.0




# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_norm(q):
    """Normalize quaternion q = [qw, qx, qy, qz]."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q


def _quat_mult(p, q):
    """Hamilton product of two quaternions [qw, qx, qy, qz]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dtype=float)


def _quat_to_rot(q):
    """Rotation matrix from quaternion [qw, qx, qy, qz] (body→world)."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ], dtype=float)


def _skew(v):
    """3×3 skew-symmetric matrix for cross-product."""
    return np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=float)


class VioAugmentedKalmanFilter:
    """
    Extended Kalman Filter with full single rigid-body dynamics.

    State (16): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
      p   (0:3)  – position in world frame (m)
      v   (3:6)  – velocity in world frame (m/s)
      q   (6:10) – orientation quaternion [qw, qx, qy, qz] (body→world)
      ba  (10:13)– accelerometer bias in body frame (m/s^2)
      bg  (13:16)– gyroscope bias in body frame (rad/s)

    Error-state covariance (15×15) uses a minimal 3-element rotation error
    parameterisation (δθ) mapped to/from the 4-element quaternion via a
    left-perturbation Jacobian, keeping the covariance well-defined.

    Index mapping for error state:
      δp  (0:3), δv  (3:6), δθ  (6:9), δba (9:12), δbg (12:15)

    IMU process model:
      ṗ = v
      v̇ = R(q)·(a_m − ba) − g
      q̇ = 0.5 · q ⊗ [0, ω_m − bg]
      ḃa = noise
      ḃg = noise
    """

    _g = np.array([0.0, 0.0, 9.81], dtype=float)  # gravity in world (ENU: +z up)

    def __init__(self, cfg: KFConfig):
        self.cfg = cfg
        # Nominal state [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
        self.x = np.zeros(16, dtype=float)
        self.x[6] = 1.0  # identity quaternion: qw=1

        # Error-state covariance (15×15)
        self.P = np.eye(15, dtype=float)

        self.t = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_state(
        self,
        pos,
        vel=(0.0, 0.0, 0.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        accel_bias=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        pos_var=1.0,
        vel_var=1.0,
        quat_var=0.1,
        accel_bias_var=0.01,
        gyro_bias_var=0.001,
        t0=0.0,
        # legacy keyword kept for back-compat with kf_node.py callers
        bias=None,
        bias_var=None,
    ):
        """Initialise nominal state and error-state covariance."""
        self.x[0:3]  = np.asarray(pos, dtype=float)
        self.x[3:6]  = np.asarray(vel, dtype=float)
        self.x[6:10] = _quat_norm(np.asarray(quat_wxyz, dtype=float))
        self.x[10:13] = np.asarray(accel_bias, dtype=float)
        self.x[13:16] = np.asarray(gyro_bias, dtype=float)

        self.P = np.diag([
            pos_var,        pos_var,        pos_var,        # δp
            vel_var,        vel_var,        vel_var,        # δv
            quat_var,       quat_var,       quat_var,       # δθ
            accel_bias_var, accel_bias_var, accel_bias_var, # δba
            gyro_bias_var,  gyro_bias_var,  gyro_bias_var,  # δbg
        ]).astype(float)

        self.t = float(t0)

    def propagate_imu(self, t_now: float, accel_meas: np.ndarray, gyro_meas: np.ndarray):
        """
        Propagate state forward using raw IMU measurements.

        Args:
            t_now:      current timestamp (s)
            accel_meas: specific force in body frame [ax, ay, az] (m/s²)
            gyro_meas:  angular velocity in body frame [wx, wy, wz] (rad/s)
        """
        if self.t is None:
            self.t = float(t_now)
            return

        dt = float(t_now - self.t)
        if dt <= 0.0:
            return

        accel_meas = np.asarray(accel_meas, dtype=float)
        gyro_meas  = np.asarray(gyro_meas,  dtype=float)

        # Bias-corrected measurements
        a_b = accel_meas - self.x[10:13]   # specific force in body
        w_b = gyro_meas  - self.x[13:16]   # angular velocity in body

        q = self.x[6:10]
        R = _quat_to_rot(q)                 # body → world

        # ---- Nominal state integration (midpoint / RK1) ----
        p  = self.x[0:3]
        v  = self.x[3:6]

        a_w = R @ a_b - self._g             # acceleration in world frame

        self.x[0:3]  = p + v*dt + 0.5*a_w*(dt**2)
        self.x[3:6]  = v + a_w*dt

        # Quaternion integration: q ← q ⊗ exp(0.5 · w_b · dt)
        angle = np.linalg.norm(w_b) * dt
        if angle > 1e-10:
            axis = w_b / (np.linalg.norm(w_b) + 1e-15)
            dq = np.concatenate([[np.cos(angle * 0.5)],
                                  np.sin(angle * 0.5) * axis])
        else:
            dq = np.array([1.0, 0.5*w_b[0]*dt, 0.5*w_b[1]*dt, 0.5*w_b[2]*dt])

        self.x[6:10] = _quat_norm(_quat_mult(q, dq))

        # biases unchanged (random walk only)

        # ---- Error-state covariance propagation ----
        # F is 15×15 (error state: δp, δv, δθ, δba, δbg)
        F = self._build_F(R, a_b, w_b, dt)
        Q = self._build_Q(dt)

        self.P = F @ self.P @ F.T + Q
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self.t = float(t_now)

    def process_event(self, event, accel_meas=None, gyro_meas=None):
        """
        Process a measurement event.  Before the measurement update the state
        is propagated to the event time using the supplied (or last stored) IMU.

        Args:
            event: dict with keys:
                   - "t":   timestamp (s)
                   - "type": "vio" | "pnp"
                   - "pos": position [x,y,z]
                   - "vel": velocity [vx,vy,vz]  (VIO only)
            accel_meas: body-frame accelerometer reading [ax,ay,az] (m/s²)
            gyro_meas:  body-frame gyroscope reading [wx,wy,wz] (rad/s)
        """
        t = float(event["t"])

        # Use zero gyro/accel if nothing provided (static assumption)
        a = np.asarray(accel_meas, dtype=float) if accel_meas is not None \
            else np.array([0.0, 0.0, 9.81])  # hover-level gravity compensation
        g = np.asarray(gyro_meas,  dtype=float) if gyro_meas  is not None \
            else np.zeros(3)

        self.propagate_imu(t, a, g)

        if event["type"] == "vio":
            self._update_vio(event["pos"], event["vel"])
        elif event["type"] == "pnp":
            self._update_pnp(event["pos"])
        else:
            raise ValueError(f"Unknown event type: {event['type']}")

    def get_state(self):
        """
        Return (position, velocity, accel_bias, covariance).

        The third return value is the accelerometer bias (previously 'VIO bias').
        The covariance is the 15×15 error-state covariance.
        """
        p  = self.x[0:3].copy()
        v  = self.x[3:6].copy()
        ba = self.x[10:13].copy()
        return p, v, ba, self.P.copy()

    def get_full_state(self):
        """Return full 16-element nominal state."""
        return self.x.copy()

    def get_quaternion(self):
        """Return current orientation quaternion [qw, qx, qy, qz]."""
        return self.x[6:10].copy()

    def update_vio(self, vio_pos, vio_vel):
        """Measurement update with VIO pos+vel – no propagation step."""
        self._update_vio(vio_pos, vio_vel)

    def update_pnp(self, pnp_pos):
        """Measurement update with PnP position – no propagation step."""
        self._update_pnp(pnp_pos)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_F(self, R, a_b, w_b, dt):
        """
        First-order linearised state-transition matrix for the error state.
        Ordering: [δp(0:3), δv(3:6), δθ(6:9), δba(9:12), δbg(12:15)]
        """
        F = np.eye(15, dtype=float)

        # δp += δv · dt
        F[0:3, 3:6] = np.eye(3) * dt

        # δv += − R · [a_b×] · δθ · dt − R · δba · dt
        F[3:6, 6:9]  = -R @ _skew(a_b) * dt
        F[3:6, 9:12] = -R * dt

        # δθ += − [w_b×] · δθ · dt − δbg · dt
        F[6:9, 6:9]  = np.eye(3) - _skew(w_b) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        return F

    def _build_Q(self, dt):
        """
        Continuous-time process noise matrix discretised for time step dt.
        Ordering: [δp, δv, δθ, δba, δbg]
        """
        Q = np.zeros((15, 15), dtype=float)
        q_a  = (self.cfg.imu_accel_noise_std ** 2) * dt
        q_g  = (self.cfg.imu_gyro_noise_std   ** 2) * dt
        q_ba = (self.cfg.accel_bias_rw_std    ** 2) * dt
        q_bg = (self.cfg.gyro_bias_rw_std     ** 2) * dt

        for i in range(3):
            Q[3+i, 3+i]  = q_a   # velocity driven by accel noise
            Q[6+i, 6+i]  = q_g   # attitude driven by gyro noise
            Q[9+i, 9+i]  = q_ba  # accel bias random walk
            Q[12+i, 12+i] = q_bg  # gyro bias random walk

        return Q

    def _error_state_to_nominal(self, dx):
        """
        Inject the 15-element error state dx into the 16-element nominal state
        and reset the error state to zero (reset step of ESKF).
        """
        self.x[0:3]  += dx[0:3]
        self.x[3:6]  += dx[3:6]
        # Quaternion: left-multiply by small-angle rotation δq ≈ [1, δθ/2]
        dtheta = dx[6:9]
        dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
        self.x[6:10] = _quat_norm(_quat_mult(self.x[6:10].copy(), dq))
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

    def _measurement_update(self, z, h_x, H, R):
        """
        Generic EKF measurement update.

        Args:
            z:   measurement vector (m,)
            h_x: predicted measurement h(x) (m,)
            H:   Jacobian of h w.r.t. error state (m×15)
            R:   measurement noise covariance (m×m)
        """
        z   = np.asarray(z,   dtype=float).reshape(-1)
        h_x = np.asarray(h_x, dtype=float).reshape(-1)

        y = z - h_x  # innovation

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # keep state unchanged on singular S

        dx = K @ y  # (15,)

        # Update covariance (Joseph form)
        I   = np.eye(15, dtype=float)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Inject error into nominal state
        self._error_state_to_nominal(dx)

    def _update_vio(self, vio_pos, vio_vel):
        """
        VIO position + velocity measurement update.
        Measurement model: z = [p, v]  (no additional bias term;
        the IMU biases already account for drift).
        """
        z   = np.hstack([vio_pos, vio_vel])
        h_x = np.hstack([self.x[0:3], self.x[3:6]])

        H = np.zeros((6, 15), dtype=float)
        H[0:3, 0:3] = np.eye(3)   # ∂p_obs/∂δp
        H[3:6, 3:6] = np.eye(3)   # ∂v_obs/∂δv

        sigma_p = self.cfg.vio_pos_std
        sigma_v = self.cfg.vio_vel_std
        if self.cfg.vio_delay_compensation and self.cfg.vio_delay_sec > 0:
            delay_factor = 1.0 + self.cfg.vio_delay_sec * 2.0
            sigma_p *= np.sqrt(delay_factor)
            sigma_v *= np.sqrt(delay_factor)

        R = np.diag([sigma_p**2]*3 + [sigma_v**2]*3).astype(float)

        self._measurement_update(z, h_x, H, R)

    def _update_pnp(self, pnp_pos):
        """PnP position measurement update (high-confidence, no additional bias)."""
        z   = np.asarray(pnp_pos, dtype=float)
        h_x = self.x[0:3]

        H = np.zeros((3, 15), dtype=float)
        H[0:3, 0:3] = np.eye(3)   # ∂p_obs/∂δp

        R = np.diag([self.cfg.pnp_pos_std**2]*3).astype(float)

        self._measurement_update(z, h_x, H, R)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    cfg = KFConfig(
        imu_accel_noise_std=0.1,
        imu_gyro_noise_std=0.01,
        accel_bias_rw_std=0.001,
        gyro_bias_rw_std=0.0001,
        vio_pos_std=0.20,
        vio_vel_std=0.30,
        pnp_pos_std=0.03,
        vio_delay_compensation=True,
        vio_delay_sec=0.05,
    )

    kf = VioAugmentedKalmanFilter(cfg)
    kf.init_state(pos=[0, 0, 0], vel=[0, 0, 0], t0=0.0)

    events = [
        {"t": 0.05, "type": "vio", "pos": [0.10, 0.01, 0.00], "vel": [1.5,  0.1,  0.0]},
        {"t": 0.10, "type": "vio", "pos": [0.25, 0.02, 0.00], "vel": [2.0,  0.15, 0.0]},
        {"t": 1.00, "type": "pnp", "pos": [1.20, 0.00, 0.00]},
    ]

    imu_history = [
        ([1.5,  0.1,  9.81], [0.0, 0.0, 0.0]),
        ([1.4,  0.12, 9.81], [0.0, 0.0, 0.0]),
        ([0.0,  0.0,  9.81], [0.0, 0.0, 0.0]),
    ]

    for i, e in enumerate(sorted(events, key=lambda x: x["t"])):
        accel, gyro = imu_history[i] if i < len(imu_history) else ([0, 0, 9.81], [0, 0, 0])
        t0 = time.time()
        kf.process_event(e, accel_meas=accel, gyro_meas=gyro)
        p, v, ba, _ = kf.get_state()
        q = kf.get_quaternion()
        print(f"t={e['t']:.2f}  type={e['type']}")
        print(f"  pos  : {p}")
        print(f"  vel  : {v}")
        print(f"  quat : {q}  (qw,qx,qy,qz)")
        print(f"  ba   : {ba}")
        print(f"  dt   : {(time.time()-t0)*1000:.2f} ms\n")