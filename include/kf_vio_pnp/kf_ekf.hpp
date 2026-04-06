#pragma once
// =============================================================================
// ESKF – Error-State Kalman Filter with full single rigid-body dynamics.
//
// Nominal state  x  (16):  [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
// Error state   dx  (15):  [δp(0:3), δv(3:6), δθ(6:9), δba(9:12), δbg(12:15)]
//
// Quaternion convention: [qw, qx, qy, qz]   (body → world, Hamilton product)
// Gravity: g = [0, 0, 9.81] m/s²    (ENU frame, +z up)
// =============================================================================

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace kf_vio_pnp {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct KFConfig {
    double imu_accel_noise_std  = 0.1;    // m/s²
    double imu_gyro_noise_std   = 0.01;   // rad/s
    double accel_bias_rw_std    = 0.001;  // m/s² per sqrt(s)
    double gyro_bias_rw_std     = 0.0001; // rad/s per sqrt(s)

    double vio_pos_std          = 0.20;   // m
    double vio_vel_std          = 0.30;   // m/s
    double pnp_pos_std          = 0.03;   // m

    bool   vio_delay_compensation = false;
    double vio_delay_sec          = 0.0;
};

// ---------------------------------------------------------------------------
// Eigen type aliases
// ---------------------------------------------------------------------------
using Vec3  = Eigen::Vector3d;
using Vec4  = Eigen::Vector4d;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Mat3  = Eigen::Matrix3d;
using Mat15 = Eigen::Matrix<double, 15, 15>;

// ---------------------------------------------------------------------------
// Quaternion helpers  [qw, qx, qy, qz]
// ---------------------------------------------------------------------------
inline Vec4 quatNorm(const Vec4& q)
{
    double n = q.norm();
    return (n > 1e-12) ? (q / n) : q;
}

inline Vec4 quatMult(const Vec4& p, const Vec4& q)
{
    double pw = p(0), px = p(1), py = p(2), pz = p(3);
    double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    return Vec4(
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw);
}

inline Mat3 quatToRot(const Vec4& q)
{
    Vec4 qn = quatNorm(q);
    double qw = qn(0), qx = qn(1), qy = qn(2), qz = qn(3);
    Mat3 R;
    R << 1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy),
           2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx),
           2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy);
    return R;
}

inline Mat3 skew(const Vec3& v)
{
    Mat3 S;
    S <<    0,  -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
}

// ---------------------------------------------------------------------------
// VioAugmentedKalmanFilter
// ---------------------------------------------------------------------------
class VioAugmentedKalmanFilter
{
public:
    VioAugmentedKalmanFilter() { reset(); }
    explicit VioAugmentedKalmanFilter(const KFConfig& cfg) : cfg_(cfg) { reset(); }

    void setConfig(const KFConfig& cfg) { cfg_ = cfg; }

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------
    void initState(const Vec3&  pos,
                   const Vec3&  vel           = Vec3::Zero(),
                   const Vec4&  quat_wxyz     = Vec4(1,0,0,0),
                   const Vec3&  accel_bias    = Vec3::Zero(),
                   const Vec3&  gyro_bias     = Vec3::Zero(),
                   double pos_var        = 1.0,
                   double vel_var        = 1.0,
                   double quat_var       = 0.1,
                   double accel_bias_var = 0.01,
                   double gyro_bias_var  = 0.001,
                   double t0             = 0.0)
    {
        x_.segment<3>(0)  = pos;
        x_.segment<3>(3)  = vel;
        x_.segment<4>(6)  = quatNorm(quat_wxyz);
        x_.segment<3>(10) = accel_bias;
        x_.segment<3>(13) = gyro_bias;

        Vec15 d;
        d << Vec3::Constant(pos_var),
             Vec3::Constant(vel_var),
             Vec3::Constant(quat_var),
             Vec3::Constant(accel_bias_var),
             Vec3::Constant(gyro_bias_var);
        P_ = d.asDiagonal();

        t_           = t0;
        initialized_ = true;
    }

    bool   isInitialized() const { return initialized_; }
    double getTime()        const { return t_; }

    // ------------------------------------------------------------------
    // IMU propagation  (drives EKF at sensor rate)
    // ------------------------------------------------------------------
    void propagateImu(double t_now,
                      const Vec3& accel_meas,
                      const Vec3& gyro_meas)
    {
        if (!initialized_) return;
        if (t_ < 0.0) { t_ = t_now; return; }

        double dt = t_now - t_;
        if (dt <= 0.0) return;
        if (dt > 1.0)  { t_ = t_now; return; }  // sanity: skip stale jumps

        const Vec3 a_b = accel_meas - x_.segment<3>(10);   // bias-corrected accel (body)
        const Vec3 w_b = gyro_meas  - x_.segment<3>(13);   // bias-corrected gyro  (body)

        const Vec4 q = x_.segment<4>(6);
        const Mat3 R = quatToRot(q);                        // body → world

        const Vec3 p = x_.segment<3>(0);
        const Vec3 v = x_.segment<3>(3);

        // Gravity in world frame (ENU): +z up
        static const Vec3 g(0.0, 0.0, 9.81);
        const Vec3 a_w = R * a_b - g;                       // world-frame acceleration

        // --- Nominal integration (Euler) ---
        x_.segment<3>(0) = p + v*dt + 0.5*a_w*(dt*dt);
        x_.segment<3>(3) = v + a_w*dt;

        // Quaternion integration: q ← q ⊗ dq  (right-perturbation body frame)
        const double angle = w_b.norm() * dt;
        Vec4 dq;
        if (angle > 1e-10) {
            const Vec3 axis = w_b / (w_b.norm() + 1e-15);
            dq << std::cos(angle * 0.5), std::sin(angle * 0.5) * axis;
        } else {
            dq << 1.0, 0.5*w_b(0)*dt, 0.5*w_b(1)*dt, 0.5*w_b(2)*dt;
        }
        x_.segment<4>(6) = quatNorm(quatMult(q, dq));

        // biases: random-walk only (handled in Q)

        // --- Error-state covariance propagation ---
        const Mat15 F = buildF(R, a_b, w_b, dt);
        const Mat15 Q = buildQ(dt);
        P_ = F * P_ * F.transpose() + Q;
        P_ = 0.5 * (P_ + P_.transpose());  // enforce symmetry

        t_ = t_now;
    }

    // ------------------------------------------------------------------
    // Measurement updates (measurement-only, no propagation)
    // ------------------------------------------------------------------
    void updateVio(const Vec3& vio_pos, const Vec3& vio_vel)
    {
        Eigen::Matrix<double, 6, 1> z;
        z << vio_pos, vio_vel;
        Eigen::Matrix<double, 6, 1> h_x;
        h_x << x_.segment<3>(0), x_.segment<3>(3);

        Eigen::Matrix<double, 6, 15> H;
        H.setZero();
        H.block<3,3>(0,0) = Mat3::Identity();   // ∂p_obs/∂δp
        H.block<3,3>(3,3) = Mat3::Identity();   // ∂v_obs/∂δv

        double sp = cfg_.vio_pos_std;
        double sv = cfg_.vio_vel_std;
        if (cfg_.vio_delay_compensation && cfg_.vio_delay_sec > 0.0) {
            const double f = std::sqrt(1.0 + cfg_.vio_delay_sec * 2.0);
            sp *= f;
            sv *= f;
        }
        Eigen::Matrix<double, 6, 6> R_cov;
        R_cov.setZero();
        R_cov.diagonal() << sp*sp, sp*sp, sp*sp, sv*sv, sv*sv, sv*sv;

        const Eigen::Matrix<double, 6, 1> innov = (z - h_x).eval();
        measurementUpdate(innov, H, R_cov);
    }

    void updatePnp(const Vec3& pnp_pos)
    {
        Eigen::Matrix<double, 3, 1> y = pnp_pos - x_.segment<3>(0);

        Eigen::Matrix<double, 3, 15> H;
        H.setZero();
        H.block<3,3>(0,0) = Mat3::Identity();   // ∂p_obs/∂δp

        const double sp = cfg_.pnp_pos_std;
        Eigen::Matrix<double, 3, 3> R_cov = Mat3::Identity() * (sp * sp);

        const Eigen::Matrix<double, 3, 1> innov = y.eval();
        measurementUpdate(innov, H, R_cov);
    }

    // ------------------------------------------------------------------
    // State accessors
    // ------------------------------------------------------------------
    struct State {
        Vec3  pos, vel, accel_bias;
        Vec4  quat;  // [qw, qx, qy, qz]
        Mat15 cov;
    };

    State getState() const
    {
        State s;
        s.pos        = x_.segment<3>(0);
        s.vel        = x_.segment<3>(3);
        s.quat       = x_.segment<4>(6);
        s.accel_bias = x_.segment<3>(10);
        s.cov        = P_;
        return s;
    }

    Vec4 getQuaternion()  const { return x_.segment<4>(6); }
    Vec3 getPosition()    const { return x_.segment<3>(0); }
    Vec3 getVelocity()    const { return x_.segment<3>(3); }
    Vec3 getAccelBias()   const { return x_.segment<3>(10); }
    Vec3 getGyroBias()    const { return x_.segment<3>(13); }

private:
    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------
    void reset()
    {
        x_.setZero();
        x_(6) = 1.0;   // identity quaternion: qw = 1
        P_.setIdentity();
        t_           = -1.0;
        initialized_ = false;
    }

    // Linearised error-state transition matrix (15×15)
    Mat15 buildF(const Mat3& R, const Vec3& a_b, const Vec3& w_b, double dt) const
    {
        Mat15 F = Mat15::Identity();
        F.block<3,3>(0, 3) = Mat3::Identity() * dt;           // δp += δv · dt
        F.block<3,3>(3, 6) = -R * skew(a_b) * dt;             // δv += -R·[a_b×]·δθ · dt
        F.block<3,3>(3, 9) = -R * dt;                          // δv += -R·δba · dt
        F.block<3,3>(6, 6) = Mat3::Identity() - skew(w_b)*dt; // δθ += -[w_b×]·δθ · dt
        F.block<3,3>(6,12) = -Mat3::Identity() * dt;           // δθ += -δbg · dt
        return F;
    }

    // Discretised process noise matrix (15×15)
    Mat15 buildQ(double dt) const
    {
        Mat15 Q = Mat15::Zero();
        const double qa  = cfg_.imu_accel_noise_std * cfg_.imu_accel_noise_std * dt;
        const double qg  = cfg_.imu_gyro_noise_std  * cfg_.imu_gyro_noise_std  * dt;
        const double qba = cfg_.accel_bias_rw_std   * cfg_.accel_bias_rw_std   * dt;
        const double qbg = cfg_.gyro_bias_rw_std    * cfg_.gyro_bias_rw_std    * dt;
        for (int i = 0; i < 3; ++i) {
            Q(3+i,  3+i)  = qa;   // velocity ← accel noise
            Q(6+i,  6+i)  = qg;   // attitude ← gyro noise
            Q(9+i,  9+i)  = qba;  // accel bias random walk
            Q(12+i, 12+i) = qbg;  // gyro bias random walk
        }
        return Q;
    }

    // Inject 15-element error state into 16-element nominal state (ESKF reset)
    void injectErrorState(const Vec15& dx)
    {
        x_.segment<3>(0)  += dx.segment<3>(0);
        x_.segment<3>(3)  += dx.segment<3>(3);
        // Quaternion left-perturbation: δq ≈ [1, δθ/2]
        const Vec3 dth = dx.segment<3>(6);
        const Vec4 dq(1.0, 0.5*dth(0), 0.5*dth(1), 0.5*dth(2));
        x_.segment<4>(6) = quatNorm(quatMult(x_.segment<4>(6), dq));
        x_.segment<3>(10) += dx.segment<3>(9);
        x_.segment<3>(13) += dx.segment<3>(12);
    }

    // Generic EKF measurement update (Joseph form)
    // y   = innovation  (m × 1)
    // H   = Jacobian    (m × 15)
    // Rn  = noise cov   (m × m)
    template<int M>
    void measurementUpdate(const Eigen::Matrix<double, M, 1>&    y,
                           const Eigen::Matrix<double, M, 15>&   H,
                           const Eigen::Matrix<double, M, M>&    Rn)
    {
        using MatMM  = Eigen::Matrix<double, M, M>;
        using Mat15M = Eigen::Matrix<double, 15, M>;

        const MatMM  S   = H * P_ * H.transpose() + Rn;
        // Use LDLT for numerical stability (S is symmetric positive-definite)
        const auto   llt = S.ldlt();
        if (llt.info() != Eigen::Success) return;

        const Mat15M K   = P_ * H.transpose() * llt.solve(MatMM::Identity());
        const Vec15  dx  = K * y;

        // Joseph form covariance update
        const Mat15 IKH = Mat15::Identity() - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * Rn * K.transpose();
        P_ = 0.5 * (P_ + P_.transpose());

        injectErrorState(dx);
    }

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------
    KFConfig cfg_;
    Eigen::Matrix<double, 16, 1> x_;   // nominal state
    Mat15   P_;                         // error-state covariance
    double  t_;                         // current EKF time (s)
    bool    initialized_;
};

} // namespace kf_vio_pnp
