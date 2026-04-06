#pragma once
// =============================================================================
// PnPPoseCompose  –  estimates quadrotor world position from a PnP detection.
//
// Formula:  T_q_to_w = T_g_to_w  ×  inv(T_g_to_q)
//   T_g_to_w  : gate-to-world transform (from GateMap)
//   T_g_to_q  : gate-to-quadrotor transform (from PnP detection)
// =============================================================================

#include <Eigen/Dense>
#include "gate_map.hpp"

namespace kf_vio_pnp {

// 4×4 homogeneous transform helpers (yaw-only rotation + translation)
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

    /// Build T_g_to_q from the gate position detected in the quadrotor frame.
    /// gate_pos_quad: [x, y, z] position of the gate centre in the quadrotor frame.
    static Eigen::Matrix4d buildTgToQ(const Eigen::Vector3d& gate_pos_quad)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,1>(0,3) = gate_pos_quad;
        return T;
    }

    /// Compute the quadrotor world position:
    ///   T_q_to_w = T_g_to_w × inv(T_g_to_q)
    /// Returns the 3D world position  [x, y, z].
    Eigen::Vector3d computeQuadPosWorld(int gate_id,
                                        const Eigen::Matrix4d& T_g_to_q) const
    {
        const GateMap::GatePose& gp = gate_map_.get(gate_id);
        const Eigen::Matrix4d T_g_to_w = poseToTransform(gp.cx, gp.cy, gp.cz, gp.yaw_rad);
        const Eigen::Matrix4d T_q_to_w = T_g_to_w * invertRigidTransform(T_g_to_q);
        return T_q_to_w.block<3,1>(0,3);
    }

private:
    const GateMap& gate_map_;
};

} // namespace kf_vio_pnp
