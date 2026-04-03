import numpy as np
from dataclasses import dataclass


@dataclass
class KFConfig:
    # IMU noise
    imu_accel_noise_std: float = 0.1  # m/s^2
    
    # VIO bias random walk std (m/s per sqrt(s))
    vio_bias_rw_std: float = 0.005

    # VIO obs noise
    vio_pos_std: float = 0.20
    vio_vel_std: float = 0.30

    # PnP obs noise (more reliable)
    pnp_pos_std: float = 0.03
    
    # VIO delay compensation
    vio_delay_compensation: bool = False
    vio_delay_sec: float = 0.0


class VioAugmentedKalmanFilter:
    """
    Augmented state KF with VIO bias estimation using IMU acceleration input.
    State: [px, py, pz, vx, vy, vz, bx, by, bz]^T
      p: position (m)
      v: velocity (m/s)
      b: VIO position bias (m) - estimated online
    
    Process model:
      p' = p + v*dt + 0.5*(a_imu)*dt^2
      v' = v + a_imu*dt
      b' = b (random walk)

      if there is no new IMU, No process update (dt=0), 
      but we can still do measurement updates with the last state and bias.
    """

    def __init__(self, cfg: KFConfig):
        self.cfg = cfg
        self.x = np.zeros((9, 1), dtype=float)  # [p, v, b]
        self.P = np.eye(9, dtype=float)
        self.t = None
        self.last_accel_meas = np.array([0.0, 0.0, 0.0], dtype=float)

    def init_state(
        self,
        pos,
        vel=(0.0, 0.0, 0.0),
        bias=(0.0, 0.0, 0.0),
        pos_var=1.0,
        vel_var=1.0,
        bias_var=0.25,
        t0=0.0
    ):
        """Initialize filter state"""
        self.x[0:3, 0] = np.asarray(pos, dtype=float)
        self.x[3:6, 0] = np.asarray(vel, dtype=float)
        self.x[6:9, 0] = np.asarray(bias, dtype=float)

        self.P = np.diag([
            pos_var, pos_var, pos_var,
            vel_var, vel_var, vel_var,
            bias_var, bias_var, bias_var
        ]).astype(float)

        self.t = float(t0)

    def _build_F_Q(self, dt: float):
        """
        Build state transition and process noise matrices with IMU acceleration input.
        
        Process model:
          p' = p + v*dt + 0.5*a_imu*dt^2
          v' = v + a_imu*dt
          b' = b (random walk)
        
        Args:
            dt: time step (seconds)
        """
        # State transition matrix F
        F = np.eye(9, dtype=float)
        
        # Position update: p = p + v*dt
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity and bias don't directly update position through F
        # (acceleration input handled separately)

        # Process noise matrix Q
        Q = np.zeros((9, 9), dtype=float)
        
        # Acceleration measurement noise propagates to position and velocity
        q_accel = self.cfg.imu_accel_noise_std ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        
        # Position noise: (0.5*dt^2)^2 = 0.25*dt^4
        Q_p = q_accel * 0.25 * dt4
        # Velocity noise: (dt)^2 = dt^2
        Q_v = q_accel * dt2
        # Position-velocity correlation: 0.5*dt^3
        Q_pv = q_accel * 0.5 * dt3
        
        for i in range(3):
            Q[i, i] = Q_p
            Q[3+i, 3+i] = Q_v
            Q[i, 3+i] = Q_pv
            Q[3+i, i] = Q_pv
        
        # VIO bias random walk
        qb = (self.cfg.vio_bias_rw_std ** 2) * max(dt, 1e-6)
        Q[6, 6] = qb
        Q[7, 7] = qb
        Q[8, 8] = qb

        return F, Q


    def predict_with_imu(self, t_now: float, accel_meas: np.ndarray):
        """
        Predict state using IMU acceleration measurement as input.
        
        Args:
            t_now: current timestamp (seconds)
            accel_meas: measured acceleration [ax, ay, az] (m/s^2)
        """
        if self.t is None:
            self.t = float(t_now)
            self.last_accel_meas = np.asarray(accel_meas, dtype=float)
            return

        dt = float(t_now - self.t)
        if dt <= 0.0:
            return

        accel_meas = np.asarray(accel_meas, dtype=float)
        
        # Build F and Q
        F, Q = self._build_F_Q(dt)
        
        # Manual state update with IMU acceleration input
        p = self.x[0:3, 0]
        v = self.x[3:6, 0]
        b = self.x[6:9, 0]
        
        # Update using kinematics with measured acceleration
        # p' = p + v*dt + 0.5*a_imu*dt^2
        self.x[0:3, 0] = p + v * dt + 0.5 * accel_meas * (dt ** 2)
        # v' = v + a_imu*dt
        self.x[3:6, 0] = v + accel_meas * dt
        # b' = b (no change through F, only random walk in Q)
        
        # Covariance update
        self.P = F @ self.P @ F.T + Q
        
        self.t = float(t_now)
        self.last_accel_meas = accel_meas


    def _update(self, z, H, R):
        """Standard Kalman filter measurement update (Joseph form)"""
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().error("Singular matrix in Kalman gain computation")
            return
        
        self.x = self.x + K @ y

        # Joseph form covariance update for numerical stability
        I = np.eye(self.P.shape[0], dtype=float)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


    def update_vio(self, vio_pos, vio_vel):
        """
        VIO measurement update.
        Measurement model:
          z = [p_true + b, v]
          where b is the VIO position bias
        
        Args:
            vio_pos: observed position from VIO [x, y, z]
            vio_vel: observed velocity from VIO [vx, vy, vz]
        """
        # Measurement: [p_obs, v_obs]
        z = np.hstack([vio_pos, vio_vel])
        
        # H matrix maps state to measurement
        H = np.zeros((6, 9), dtype=float)
        H[0:3, 0:3] = np.eye(3)  # p (true position)
        H[0:3, 6:9] = np.eye(3)  # + b (VIO bias)
        H[3:6, 3:6] = np.eye(3)  # v (velocity)
        
        # Measurement covariance
        R = np.diag([
            self.cfg.vio_pos_std**2, self.cfg.vio_pos_std**2, self.cfg.vio_pos_std**2,
            self.cfg.vio_vel_std**2, self.cfg.vio_vel_std**2, self.cfg.vio_vel_std**2
        ]).astype(float)

        # Increase noise if VIO delay compensation enabled
        if self.cfg.vio_delay_compensation and self.cfg.vio_delay_sec > 0:
            delay_factor = 1.0 + (self.cfg.vio_delay_sec * 2.0)
            R = R * delay_factor
            
        self._update(z, H, R)


    def update_pnp(self, pnp_pos):
        """
        PnP measurement update (high-confidence position).
        Measurement model:
          z = p_true (no bias in PnP)
        
        Args:
            pnp_pos: position from PnP [x, y, z]
        """
        z = np.asarray(pnp_pos, dtype=float)

        H = np.zeros((3, 9), dtype=float)
        H[0:3, 0:3] = np.eye(3)  # Only true position, no bias

        R = np.diag([
            self.cfg.pnp_pos_std**2,
            self.cfg.pnp_pos_std**2,
            self.cfg.pnp_pos_std**2
        ]).astype(float)

        self._update(z, H, R)


    def process_event(self, event, accel_meas=None):
        """
        Process measurement event with IMU acceleration input.
        
        Args:
            event: measurement event dict with keys:
                   - "t": timestamp
                   - "type": "vio" or "pnp"
                   - "pos": position measurement
                   - "vel": velocity measurement (for VIO only)
            accel_meas: IMU acceleration measurement [ax, ay, az]
        """
        t = float(event["t"])
        
        # Predict with IMU acceleration input
        if accel_meas is not None:
            self.predict_with_imu(t, accel_meas)
        else:
            self.predict_with_imu(t, self.last_accel_meas)

        # Update with measurement
        if event["type"] == "vio":
            self.update_vio(event["pos"], event["vel"])
        elif event["type"] == "pnp":
            self.update_pnp(event["pos"])
        else:
            raise ValueError(f"Unknown event type: {event['type']}")

    def get_state(self):
        """Get current state estimate: position, velocity, VIO bias, covariance"""
        p = self.x[0:3, 0].copy()
        v = self.x[3:6, 0].copy()
        b = self.x[6:9, 0].copy()
        return p, v, b, self.P.copy()


if __name__ == "__main__":
    cfg = KFConfig(
        imu_accel_noise_std=0.1,
        vio_bias_rw_std=0.005,
        vio_pos_std=0.20,
        vio_vel_std=0.30,
        pnp_pos_std=0.03,
        vio_delay_compensation=True,
        vio_delay_sec=0.05
    )

    kf = VioAugmentedKalmanFilter(cfg)
    kf.init_state(
        pos=[0, 0, 0],
        vel=[0, 0, 0],
        bias=[0, 0, 0],
        t0=0.0
    )

    # test
    events = [
        {
            "t": 0.05,
            "type": "vio",
            "pos": [0.10, 0.01, 0.00],
            "vel": [1.5, 0.1, 0.0],
        },
        {
            "t": 0.10,
            "type": "vio",
            "pos": [0.25, 0.02, 0.00],
            "vel": [2.0, 0.15, 0.0],
        },
        {
            "t": 1.00,
            "type": "pnp",
            "pos": [1.20, 0.00, 0.00]
        },
    ]

    import time
    imu_accel_history = [
        [1.5, 0.1, 0.0],
        [1.4, 0.12, 0.0],
        [0.0, 0.0, 0.0],
    ]

    for i, e in enumerate(sorted(events, key=lambda x: x["t"])):
        accel_meas = imu_accel_history[i] if i < len(imu_accel_history) else [0, 0, 0]
        
        start_time = time.time()
        kf.process_event(e, accel_meas=accel_meas)
        p, v, b, _ = kf.get_state()
        end_time = time.time()
        time_used = end_time - start_time
        
        print(f"t={e['t']:.2f}, type={e['type']}")
        print(f"  IMU accel: {accel_meas}")
        print(f"  Position: {p}")
        print(f"  Velocity: {v}")
        print(f"  VIO bias: {b}")
        print(f"  Time: {time_used*1000:.2f}ms\n")