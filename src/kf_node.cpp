// =============================================================================
// kf_node.cpp  –  ROS 1 EKF fusion node (C++ port)
//
// IMU / VIO time-alignment strategy
// ─────────────────────────────────
//  • IMU callback  : filter raw data, then push the sample into a timestamped
//                    ring buffer.  No EKF propagation happens here.
//  • VIO / PnP callbacks : call drainBufferTo(t_meas), which pops every
//                    buffered IMU sample whose timestamp ≤ t_meas and runs
//                    EKF propagation in strict chronological order.
//                    After all matching IMU samples are consumed the
//                    measurement update is applied at exactly t_meas.
//  • Result        : the EKF state is always propagated to the exact
//                    measurement timestamp before any update, eliminating
//                    the IMU/VIO time-alignment error.
//
// Thread safety
// ─────────────
//  All accesses to the EKF, the IMU buffer, and the loggers are serialised
//  by a single std::mutex (ekf_mutex_).  ROS callbacks are dispatched from
//  a multi-thread async spinner (N = 4 threads), so the mutex is essential.
// =============================================================================

#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/Imu.h>

#include <Eigen/Dense>

#include <deque>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <filesystem>   // C++17

#include "kf_vio_pnp/kf_ekf.hpp"
#include "kf_vio_pnp/butterworth_filter.hpp"
#include "kf_vio_pnp/gate_map.hpp"
#include "kf_vio_pnp/pnp_pose_compose.hpp"

using namespace kf_vio_pnp;

// ============================================================
// Utility: current-time timestamp string
// ============================================================
static std::string nowTimestamp()
{
    std::time_t t  = std::time(nullptr);
    std::tm*    tm = std::localtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", tm);
    return std::string(buf);
}

// ============================================================
// CsvLogger  –  generic timestamped CSV file logger
// ============================================================
class CsvLogger
{
public:
    CsvLogger(const std::string& dir,
              const std::string& prefix,
              const std::vector<std::string>& columns)
    {
        // Create directory
        std::filesystem::create_directories(dir);
        path_ = dir + "/" + prefix + "_" + nowTimestamp() + ".csv";
        f_.open(path_);
        if (!f_.is_open())
            throw std::runtime_error("CsvLogger: cannot open " + path_);

        // Write header
        for (std::size_t i = 0; i < columns.size(); ++i) {
            if (i) f_ << ",";
            f_ << columns[i];
        }
        f_ << "\n";
        ROS_INFO("CsvLogger: writing %s", path_.c_str());
    }

    ~CsvLogger() { close(); }

    template<typename... Args>
    void write(Args&&... args)
    {
        writeImpl(f_, std::forward<Args>(args)...);
        f_ << "\n";
        f_.flush();
    }

    void close()
    {
        if (f_.is_open()) {
            f_.close();
            ROS_INFO("CsvLogger: closed %s", path_.c_str());
        }
    }

    const std::string& path() const { return path_; }

private:
    std::ofstream f_;
    std::string   path_;

    void writeImpl(std::ofstream&) {}   // base-case (0 arguments)

    template<typename T>
    void writeImpl(std::ofstream& os, T&& v)
    {
        os << std::forward<T>(v);
    }

    template<typename T, typename... Rest>
    void writeImpl(std::ofstream& os, T&& v, Rest&&... rest)
    {
        os << std::forward<T>(v) << ",";
        writeImpl(os, std::forward<Rest>(rest)...);
    }
};

// Variadic CSV row helper (comma-separated via fold expression)
template<typename... Args>
void csvWrite(CsvLogger& logger, Args&&... args) {
    logger.write(std::forward<Args>(args)...);
}

// ============================================================
// Transform  –  VIO frame to PnP (world) frame
// ============================================================
class Transform
{
public:
    Transform(double yaw_rad, const Eigen::Vector3d& translation)
        : yaw_(yaw_rad), t_(translation)
    {
        const double c = std::cos(yaw_);
        const double s = std::sin(yaw_);
        R_ << c, -s, 0,
              s,  c, 0,
              0,  0, 1;
    }

    Eigen::Vector3d vioPositionToWorld(const Eigen::Vector3d& p) const
    {
        return R_ * p + t_;
    }

    Eigen::Vector3d vioVelocityToWorld(const Eigen::Vector3d& v) const
    {
        return R_ * v;
    }

    // Transform VIO quaternion [x,y,z,w] to PnP quaternion [x,y,z,w]
    // The yaw offset is applied as a left-multiplication: q_pnp = q_yaw ⊗ q_vio
    Eigen::Vector4d vioQuatToWorld(const Eigen::Vector4d& q_xyzw_vio) const
    {
        // Convert VIO quat from ROS [x,y,z,w] to internal [qw,qx,qy,qz]
        Vec4 qv(q_xyzw_vio(3), q_xyzw_vio(0), q_xyzw_vio(1), q_xyzw_vio(2));

        // Yaw-offset quaternion (half-angle rotation around z)
        Vec4 q_off(std::cos(yaw_*0.5), 0.0, 0.0, std::sin(yaw_*0.5));

        Vec4 q_pnp = quatNorm(quatMult(q_off, qv));
        // Return as [qw, qx, qy, qz] (internal convention)
        return q_pnp;
    }

    const Eigen::Matrix3d& R() const { return R_; }

private:
    double          yaw_;
    Eigen::Vector3d t_;
    Eigen::Matrix3d R_;
};

// ============================================================
// MocapGatekeeper  – downsampling / periodic gating
// ============================================================
class MocapGatekeeper
{
public:
    MocapGatekeeper(const std::string& mode,
                    double freq,
                    double available_dur,
                    double unavailable_dur)
        : mode_(mode)
        , freq_(freq)
        , avail_dur_(available_dur)
        , unavail_dur_(unavailable_dur)
    {}

    bool shouldSkip(double t)
    {
        if (mode_ == "continuous") return false;

        if (mode_ == "downsampled") {
            ++count_;
            if (count_ > 10000) count_ = 1;
            const int skip = std::max(1, static_cast<int>(100.0 / freq_));
            return (count_ % skip) != 0;
        }

        if (mode_ == "periodic") {
            if (!t_init_set_) { t_init_ = t; t_init_set_ = true; }
            const double cycle = avail_dur_ + unavail_dur_;
            const double pos   = std::fmod(t - t_init_, cycle);
            if (pos >= avail_dur_) return true;

            ++count_;
            if (count_ > 10000) count_ = 1;
            const int skip = std::max(1, static_cast<int>(100.0 / freq_));
            return (count_ % skip) != 0;
        }

        ROS_WARN_ONCE("MocapGatekeeper: unknown mode '%s', passing all", mode_.c_str());
        return false;
    }

private:
    std::string mode_;
    double      freq_;
    double      avail_dur_;
    double      unavail_dur_;
    int         count_      = 0;
    double      t_init_     = 0.0;
    bool        t_init_set_ = false;
};

// ============================================================
// KFNode
// ============================================================
class KFNode
{
    // ----------------------------------------------------------
    // Buffered IMU sample
    // ----------------------------------------------------------
    struct ImuSample {
        double          t;
        Eigen::Vector3d accel;   // filtered, body frame (m/s²)
        Eigen::Vector3d gyro;    // filtered, body frame (rad/s)
    };

    static constexpr std::size_t IMU_BUFFER_MAX = 2000;   // ~10 s @ 200 Hz

public:
    KFNode()
    {
        ros::NodeHandle nh("~");

        // ── KF config ─────────────────────────────────────────
        KFConfig cfg;
        cfg.imu_accel_noise_std  = nh.param("imu_accel_noise_std",  0.1);
        cfg.imu_gyro_noise_std   = nh.param("imu_gyro_noise_std",   0.01);
        cfg.accel_bias_rw_std    = nh.param("accel_bias_rw_std",    0.001);
        cfg.gyro_bias_rw_std     = nh.param("gyro_bias_rw_std",     0.0001);
        cfg.vio_pos_std          = nh.param("vio_pos_std",          0.20);
        cfg.vio_vel_std          = nh.param("vio_vel_std",          0.30);
        cfg.vio_quat_std         = nh.param("vio_quat_std",         0.05);
        cfg.pnp_pos_std          = nh.param("pnp_pos_std",          0.03);
        cfg.vio_delay_compensation  = nh.param("vio_delay_compensation",  true);
        cfg.vio_delay_sec           = nh.param("vio_delay_sec",           0.05);
        cfg.vio_pos_bias_rw_std     = nh.param("vio_pos_bias_rw_std",     0.001);

        kf_.setConfig(cfg);

        gate_pose_delay_sec_ = nh.param("gate_pose_delay_sec", 0.1);
        pnp_gate_sigma_       = nh.param("pnp_gate_sigma",      5.0);
        ROS_INFO("Gate pose delay compensation: %.3f s", gate_pose_delay_sec_);
        ROS_INFO("PnP Mahalanobis gate: %.1f sigma", pnp_gate_sigma_);

        // ── Frame transform (VIO to world/PnP) ─────────────────
        Eigen::Vector3d init_pos(
            nh.param("init_pos_x", 0.0),
            nh.param("init_pos_y", 0.0),
            nh.param("init_pos_z", 0.0));
        const double init_yaw = nh.param("init_yaw_rad", 0.0);
        transform_ = std::make_unique<Transform>(init_yaw, init_pos);

        ROS_INFO_STREAM("VIOtoWorld R:\n" << transform_->R());

        // ── MoCap gatekeeper ──────────────────────────────────
        gatekeeper_ = std::make_unique<MocapGatekeeper>(
            nh.param<std::string>("mocap_mode",             "continuous"),
            nh.param("mocap_freq",                          100.0),
            nh.param("mocap_available_duration",            2.0),
            nh.param("mocap_unavailable_duration",          5.0));

        // ── Gate map & PnP composer ───────────────────────────
        std::string scene_csv = nh.param<std::string>("scene_csv_path", "config/scene.csv");
        ROS_INFO("scene_csv_path: %s", scene_csv.c_str());
        if (scene_csv[0] != '/') {
            // Resolve relative to package directory (one level above src/)
            const std::string pkg_dir =
                ros::package::getPath("kf_vio_pnp");
            scene_csv = pkg_dir + "/" + scene_csv;
        }
        gate_map_     = std::make_unique<GateMap>(scene_csv);
        pnp_compose_  = std::make_unique<PnPPoseCompose>(*gate_map_);

        // ── Butterworth filters ───────────────────────────────
        const double imu_hz        = nh.param("imu_sample_hz",       200.0);
        const double accel_cutoff  = nh.param("imu_accel_cutoff_hz",  30.0);
        const double gyro_cutoff   = nh.param("imu_gyro_cutoff_hz",   50.0);
        accel_filter_ = std::make_unique<OnlineButterworthFilter>(accel_cutoff, imu_hz);
        gyro_filter_  = std::make_unique<OnlineButterworthFilter>(gyro_cutoff,  imu_hz);

        // ── Log directory ─────────────────────────────────────
        log_dir_ = ros::package::getPath("kf_vio_pnp") + "/data/logs";

        // ── Subscribers ───────────────────────────────────────
        sub_imu_  = nh_.subscribe("/mavros/imu/data_raw", 200,
                                  &KFNode::imuCallback,   this);
        sub_vio_  = nh_.subscribe("/d2vins/odometry",       5,
                                  &KFNode::vioCallback,   this);
        sub_gate_ = nh_.subscribe("/gate_pose/pose",         5,
                                  &KFNode::gatePoseCallback, this);
        sub_mocap_= nh_.subscribe("/mavros/vision_pose/pose", 5,
                                  &KFNode::mocapCallback,  this);

        // ── Publisher ─────────────────────────────────────────
        pub_fused_ = nh_.advertise<nav_msgs::Odometry>("/kf_vio_pnp/odometry", 5);

        ROS_INFO("KFNode (C++) initialised.");
    }

    void spin()
    {
        // Use 4 threads so IMU (high-rate) doesn't starve VIO callbacks
        ros::AsyncSpinner spinner(4);
        spinner.start();
        ros::waitForShutdown();
    }

private:
    // ──────────────────────────────────────────────────────────
    // IMU callback  –  filter + buffer ONLY, no EKF propagation
    // ──────────────────────────────────────────────────────────
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
    {
        const double t = msg->header.stamp.toSec();

        Eigen::Vector3d raw_accel(msg->linear_acceleration.x,
                                  msg->linear_acceleration.y,
                                  msg->linear_acceleration.z);
        Eigen::Vector3d raw_gyro(msg->angular_velocity.x,
                                 msg->angular_velocity.y,
                                 msg->angular_velocity.z);

        // Apply Butterworth LPF (thread-safe: filters have their own state,
        // but are only called from IMU callback so no lock needed here)
        Eigen::Vector3d fa = accel_filter_->filterVec3(raw_accel);
        Eigen::Vector3d fg = gyro_filter_->filterVec3(raw_gyro);

        {
            std::lock_guard<std::mutex> lk(ekf_mutex_);

            // Reject out-of-order or duplicated samples
            if (!imu_buffer_.empty() && t <= imu_buffer_.back().t)
                return;

            imu_buffer_.push_back({t, fa, fg});

            // Bound buffer size to avoid unbounded growth during long outages
            while (imu_buffer_.size() > IMU_BUFFER_MAX)
                imu_buffer_.pop_front();
        }
    }

    // ──────────────────────────────────────────────────────────
    // VIO callback  –  drain IMU buffer to t_vio, then update
    // ──────────────────────────────────────────────────────────
    void vioCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        const double t = msg->header.stamp.toSec();

        // Extract VIO measurements
        Eigen::Vector3d pos_vio(msg->pose.pose.position.x,
                                msg->pose.pose.position.y,
                                msg->pose.pose.position.z);
        Eigen::Vector3d vel_vio(msg->twist.twist.linear.x,
                                msg->twist.twist.linear.y,
                                msg->twist.twist.linear.z);
        const auto& ori = msg->pose.pose.orientation;
        // VIO quat in ROS [x,y,z,w] order
        Eigen::Vector4d q_vio_xyzw(ori.x, ori.y, ori.z, ori.w);

        // Transform to world frame
        const Eigen::Vector3d pos_w = transform_->vioPositionToWorld(pos_vio);
        const Eigen::Vector3d vel_w = transform_->vioVelocityToWorld(vel_vio);
        // q_wxyz in internal convention
        const Vec4 q_wxyz_w         = transform_->vioQuatToWorld(q_vio_xyzw);
        std::lock_guard<std::mutex> lk(ekf_mutex_);

        if (!kf_.isInitialized()) {
            initFilter(t, pos_w, vel_w, q_wxyz_w);
            drainBufferTo(t);   // consume any pre-init buffered IMU
            return;
        }

        // ── Strict time alignment: propagate EKF to exactly t ──
        drainBufferTo(t);

        // ── Measurement update: position + velocity + orientation ──
        kf_.updateVioWithOrientation(pos_w, vel_w, q_wxyz_w);

        // VIO logger (world-frame position, velocity, orientation)
        if (log_vio_) {
            log_vio_->write(t,
                pos_w(0), pos_w(1), pos_w(2),
                vel_w(0), vel_w(1), vel_w(2),
                q_wxyz_w(0), q_wxyz_w(1), q_wxyz_w(2), q_wxyz_w(3));  // qw,qx,qy,qz
        }

        publishFused(t);
        logState(t);
    }

    // ──────────────────────────────────────────────────────────
    // Gate-pose callback  –  PnP from gate detection
    // ──────────────────────────────────────────────────────────
    void gatePoseCallback(const geometry_msgs::PoseArray::ConstPtr& msg)
    {
        // Subtract the known processing delay to get the true measurement time
        const double t_ros    = msg->header.stamp.toSec();
        const double t_actual = t_ros - gate_pose_delay_sec_;

        GateDetection det;
        if (!GatePoseArrayDecoder::decode(msg->poses, det)) return;

        ROS_INFO_THROTTLE(1.0, "Gate %d (%s), quad-frame pos [%.2f %.2f %.2f], delay-comp t=%.3f",
            det.gate_id, det.is_front ? "front" : "back",
            det.position(0), det.position(1), det.position(2), t_actual);

        const Eigen::Matrix4d T_g_to_q = PnPPoseCompose::buildTgToQ(det.position);
        const Eigen::Vector3d quad_pos  = pnp_compose_->computeQuadPosWorld(det.gate_id, T_g_to_q);

        handlePnpMeasurement(t_actual, quad_pos);
    }

    // ──────────────────────────────────────────────────────────
    // MoCap callback  –  direct position as PnP
    // ──────────────────────────────────────────────────────────
    void mocapCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        const double t = msg->header.stamp.toSec();
        Eigen::Vector3d pos(msg->pose.position.x,
                            msg->pose.position.y,
                            msg->pose.position.z);

        if (gatekeeper_->shouldSkip(t)) return;
        handlePnpMeasurement(t, pos);
    }

    // ──────────────────────────────────────────────────────────
    // Shared PnP measurement logic
    // ──────────────────────────────────────────────────────────
    void handlePnpMeasurement(double t, const Eigen::Vector3d& pos)
    {
        std::lock_guard<std::mutex> lk(ekf_mutex_);

        if (!kf_.isInitialized()) {
            initFilter(t, pos, Eigen::Vector3d::Zero(), Vec4(1,0,0,0));
            drainBufferTo(t);
            return;
        }

        drainBufferTo(t);

        const bool accepted = kf_.updatePnp(pos, pnp_gate_sigma_);

        if (accepted) {
            if (log_pnp_)
                log_pnp_->write(t, pos(0), pos(1), pos(2));
            // PnP update: correct state only, do not publish or log kf_traj
            // (the corrected state will be logged by the next VIO callback)
        } else {
            ROS_WARN_THROTTLE(1.0, "PnP measurement rejected by Mahalanobis gate (sigma=%.1f)",
                              pnp_gate_sigma_);
            if (log_pnp_)
                log_pnp_->write(t, pos(0), pos(1), pos(2));
            // PnP update: correct state only, do not publish or log kf_traj
            // (the corrected state will be logged by the next VIO callback)
        }
    }

    // ──────────────────────────────────────────────────────────
    // Core: drain IMU buffer and propagate EKF up to t_target
    //
    // Called with ekf_mutex_ HELD.
    // ──────────────────────────────────────────────────────────
    void drainBufferTo(double t_target)
    {
        while (!imu_buffer_.empty() && imu_buffer_.front().t <= t_target) {
            const ImuSample& s = imu_buffer_.front();
            kf_.propagateImu(s.t, s.accel, s.gyro);
            last_accel_ = s.accel;
            last_gyro_  = s.gyro;
            imu_buffer_.pop_front();
        }

        // Edge case: measurement arrived before any IMU (sensor startup gap)
        // or after a long IMU outage. Bridge the gap with the last known IMU.
        if (kf_.isInitialized() && kf_.getTime() < t_target - 1e-5) {
            kf_.propagateImu(t_target, last_accel_, last_gyro_);
        }
    }

    // ──────────────────────────────────────────────────────────
    // Filter initialisation  (called with ekf_mutex_ HELD)
    // ──────────────────────────────────────────────────────────
    void initFilter(double t,
                    const Eigen::Vector3d& pos,
                    const Eigen::Vector3d& vel,
                    const Vec4& quat_wxyz)
    {
        kf_.initState(pos, vel, quat_wxyz,
                      /*accel_bias=*/Eigen::Vector3d::Zero(),
                      /*gyro_bias =*/Eigen::Vector3d::Zero(),
                      /*pos_var=*/1.0, /*vel_var=*/1.0,
                      /*quat_var=*/0.1, /*accel_bias_var=*/0.01,
                      /*gyro_bias_var=*/0.001,
                      /*vio_bias_var=*/4.0,   // large initial uncertainty lets filter discover bias quickly
                      /*t0=*/t);

        // Set up loggers
        try {
            log_bias_ = std::make_unique<CsvLogger>(
                log_dir_, "bias_data",
                std::vector<std::string>{"timestamp","bias_x","bias_y","bias_z"});
            log_vio_ = std::make_unique<CsvLogger>(
                log_dir_, "vio_traj",
                std::vector<std::string>{"timestamp","px","py","pz","vx","vy","vz","qw","qx","qy","qz"});
            log_kf_ = std::make_unique<CsvLogger>(
                log_dir_, "kf_traj",
                std::vector<std::string>{"timestamp","px","py","pz","vx","vy","vz","qw","qx","qy","qz"});
            log_pnp_ = std::make_unique<CsvLogger>(
                log_dir_, "pnp_detections",
                std::vector<std::string>{"timestamp","px","py","pz"});
            log_vio_bias_ = std::make_unique<CsvLogger>(
                log_dir_, "vio_pos_bias",
                std::vector<std::string>{"timestamp","bvx","bvy","bvz"});
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to create loggers: %s", e.what());
        }

        ROS_INFO("KF initialised at t=%.3f  pos=[%.2f %.2f %.2f]",
                 t, pos(0), pos(1), pos(2));
    }

    // ──────────────────────────────────────────────────────────
    // Publish fused odometry  (called with ekf_mutex_ HELD)
    // ──────────────────────────────────────────────────────────
    void publishFused(double t)
    {
        const Vec4 q_wxyz = kf_.getQuaternion();   // [qw, qx, qy, qz]
        const Vec3 pos    = kf_.getPosition();
        const Vec3 vel    = kf_.getVelocity();

        nav_msgs::Odometry msg;
        msg.header.stamp    = ros::Time(t);
        msg.header.frame_id = "map";
        msg.child_frame_id  = "base_link";

        msg.pose.pose.position.x    = pos(0);
        msg.pose.pose.position.y    = pos(1);
        msg.pose.pose.position.z    = pos(2);
        msg.pose.pose.orientation.w = q_wxyz(0);   // qw
        msg.pose.pose.orientation.x = q_wxyz(1);   // qx
        msg.pose.pose.orientation.y = q_wxyz(2);   // qy
        msg.pose.pose.orientation.z = q_wxyz(3);   // qz

        msg.twist.twist.linear.x = vel(0);
        msg.twist.twist.linear.y = vel(1);
        msg.twist.twist.linear.z = vel(2);

        pub_fused_.publish(msg);
    }

    // ──────────────────────────────────────────────────────────
    // Log EKF state  (called with ekf_mutex_ HELD)
    // ──────────────────────────────────────────────────────────
    void logState(double t)
    {
        const Vec3 p   = kf_.getPosition();
        const Vec3 v   = kf_.getVelocity();
        const Vec3 ba  = kf_.getAccelBias();
        const Vec4 q   = kf_.getQuaternion();   // [qw, qx, qy, qz]
        const Vec3 bv  = kf_.getVioPosBias();   // world-frame VIO position bias

        if (log_bias_)
            log_bias_->write(t, ba(0), ba(1), ba(2));

        if (log_kf_)
            log_kf_->write(t, p(0), p(1), p(2), v(0), v(1), v(2),
                           q(0), q(1), q(2), q(3));

        if (log_vio_bias_)
            log_vio_bias_->write(t, bv(0), bv(1), bv(2));
    }

    // ──────────────────────────────────────────────────────────
    // Member data
    // ──────────────────────────────────────────────────────────
    ros::NodeHandle nh_;

    // EKF and its serialisation lock
    VioAugmentedKalmanFilter     kf_;
    std::mutex                   ekf_mutex_;

    // IMU ring buffer (filled by IMU callback, drained by measurement callbacks)
    std::deque<ImuSample>        imu_buffer_;
    Eigen::Vector3d              last_accel_{0.0, 0.0, 9.81};
    Eigen::Vector3d              last_gyro_ {0.0, 0.0, 0.0};

    // Sensor pipeline helpers
    std::unique_ptr<Transform>            transform_;
    std::unique_ptr<MocapGatekeeper>      gatekeeper_;
    std::unique_ptr<GateMap>              gate_map_;
    std::unique_ptr<PnPPoseCompose>       pnp_compose_;
    std::unique_ptr<OnlineButterworthFilter> accel_filter_;
    std::unique_ptr<OnlineButterworthFilter> gyro_filter_;

    // Gate pose delay compensation (subtracted from header stamp before EKF use)
    double gate_pose_delay_sec_ = 0.1;
    // Mahalanobis distance gate for PnP (reject if sqrt(y^T S^-1 y) > sigma)
    double pnp_gate_sigma_ = 5.0;

    // ROS interface
    ros::Subscriber sub_imu_, sub_vio_, sub_gate_, sub_mocap_;
    ros::Publisher  pub_fused_;

    // CSV loggers (created on EKF init)
    std::string                  log_dir_;
    std::unique_ptr<CsvLogger>   log_bias_;
    std::unique_ptr<CsvLogger>   log_vio_;
    std::unique_ptr<CsvLogger>   log_kf_;
    std::unique_ptr<CsvLogger>   log_pnp_;
    std::unique_ptr<CsvLogger>   log_vio_bias_;   // estimated VIO position bias (world frame)
};

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "kf_node");

    try {
        KFNode node;
        node.spin();
    } catch (const std::exception& e) {
        ROS_FATAL("KFNode fatal error: %s", e.what());
        return 1;
    }
    return 0;
}
