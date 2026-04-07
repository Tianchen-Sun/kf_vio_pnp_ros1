#pragma once
// =============================================================================
// PnPPoseCompose  –  estimates quadrotor world pose from a PnP detection.
//
// Position formula:  T_q_to_w = T_g_to_w  ×  inv(T_g_to_q)
//   T_g_to_w  : gate-to-world transform (from GateMap, yaw-only rotation)
//   T_g_to_q  : gate-to-quadrotor transform (from PnP detection, full rotation + translation)
//
// Orientation result:
//   R_q_to_w = R_g_to_w * R_g_to_q^T
//   Returned as quaternion [qw, qx, qy, qz] representing the quadrotor orientation in world frame.
// =============================================================================

#include <Eigen/Dense>
#include "gate_map.hpp"

namespace kf_vio_pnp {

// ---------------------------------------------------------------------------
// Quaternion / rotation matrix helpers
// ---------------------------------------------------------------------------

/// Build a 3×3 rotation matrix from a quaternion [qw, qx, qy, qz].
inline Eigen::Matrix3d quatToRotMat(double qw, double qx, double qy, double qz)
{
    const double n = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (n < 1e-10) return Eigen::Matrix3d::Identity();
    const double w = qw/n, x = qx/n, y = qy/n, z = qz/n;
    Eigen::Matrix3d R;
    R << 1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y),
         2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x),
         2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y);
    return R;
}

/// Convert a 3×3 rotation matrix to a quaternion [qw, qx, qy, qz] (Shepperd method).
inline Eigen::Vector4d rotMatToQuat(const Eigen::Matrix3d& R)
{
    double qw, qx, qy, qz;
    const double tr = R.trace();
    if (tr > 0.0) {
        const double s = 2.0 * std::sqrt(tr + 1.0);
        qw = 0.25 * s;
        qx = (R(2,1) - R(1,2)) / s;
        qy = (R(0,2) - R(2,0)) / s;
        qz = (R(1,0) - R(0,1)) / s;
    } else if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2));
        qw = (R(2,1) - R(1,2)) / s;
        qx = 0.25 * s;
        qy = (R(0,1) + R(1,0)) / s;
        qz = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2));
        qw = (R(0,2) - R(2,0)) / s;
        qx = (R(0,1) + R(1,0)) / s;
        qy = 0.25 * s;
        qz = (R(1,2) + R(2,1)) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1));
        qw = (R(1,0) - R(0,1)) / s;
        qx = (R(0,2) + R(2,0)) / s;
        qy = (R(1,2) + R(2,1)) / s;
        qz = 0.25 * s;
    }
    return Eigen::Vector4d(qw, qx, qy, qz).normalized();
}

// ---------------------------------------------------------------------------
// 4×4 homogeneous transform helpers (yaw-only rotation + translation)
// ---------------------------------------------------------------------------
inline Eigen::Matrix4d poseToTransform(double cx, double cy, double cz, double yaw_rad)
{
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0,0) =  c;  T(0,1) = -s;
    T(1,0) =  s;  T(1,1) =  c;
    T(0,3) = cx;
    T(1,3) = cy;
    T(2,3) = cz;
    return T;
}

inline Eigen::Matrix4d invertRigidTransform(const Eigen::Matrix4d& T)
{
    Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
    Tinv.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
    Tinv.block<3,1>(0,3) = -T.block<3,3>(0,0).transpose() * T.block<3,1>(0,3);
    return Tinv;
}

class PnPPoseCompose
{
public:
    explicit PnPPoseCompose(const GateMap& gate_map) : gate_map_(gate_map) {}

    /// Build T_g_to_q from the gate position and orientation detected in the quadrotor frame.
    /// gate_pos_quad:  [x, y, z] of the gate centre in the quadrotor frame.
    /// gate_quat_wxyz: quaternion [qw,qx,qy,qz] representing R_gate_to_quad (from PnP).
    ///                 Defaults to identity when orientation is not available.
    static Eigen::Matrix4d buildTgToQ(const Eigen::Vector3d& gate_pos_quad,
                                      const Eigen::Vector4d& gate_quat_wxyz =
                                          Eigen::Vector4d(1.0, 0.0, 0.0, 0.0))
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = quatToRotMat(gate_quat_wxyz(0), gate_quat_wxyz(1),
                                          gate_quat_wxyz(2), gate_quat_wxyz(3));
        T.block<3,1>(0,3) = gate_pos_quad;
        return T;
    }

    /// Combined result: quadrotor world position and orientation.
    struct QuadPose {
        Eigen::Vector3d position;         // [x, y, z] in world frame
        Eigen::Vector4d orientation_wxyz;  // quaternion [qw,qx,qy,qz]: R_quad_to_world
    };

    /// Compute the quadrotor world position:
    ///   T_q_to_w = T_g_to_w × inv(T_g_to_q)
    /// Returns the 3D world position  [x, y, z].
    Eigen::Vector3d computeQuadPosWorld(int gate_id,
                                        const Eigen::Matrix4d& T_g_to_q) const
    {
        return computeQuadPoseWorld(gate_id, T_g_to_q).position;
    }

    /// Compute both the quadrotor world position AND orientation:
    ///   T_q_to_w = T_g_to_w × inv(T_g_to_q)
    ///   position        = T_q_to_w.translation
    ///   orientation     = rotMatToQuat(T_q_to_w.rotation)   → R_quad_to_world
    QuadPose computeQuadPoseWorld(int gate_id,
                                  const Eigen::Matrix4d& T_g_to_q) const
    {
        const GateMap::GatePose& gp = gate_map_.get(gate_id);
        const Eigen::Matrix4d T_g_to_w = poseToTransform(gp.cx, gp.cy, gp.cz, gp.yaw_rad);
        const Eigen::Matrix4d T_q_to_w = T_g_to_w * invertRigidTransform(T_g_to_q);
        QuadPose result;
        result.position         = T_q_to_w.block<3,1>(0,3);
        result.orientation_wxyz = rotMatToQuat(T_q_to_w.block<3,3>(0,0));
        return result;
    }

private:
    const GateMap& gate_map_;
};

} // namespace kf_vio_pnp
