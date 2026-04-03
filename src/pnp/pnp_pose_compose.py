from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gate_map import GateMap


@dataclass(frozen=True)
class PnPPoseComposeResult:
    gate_id: str
    gate_pose_world: np.ndarray # (cx, cy, cz, yaw_rad)
    T_g_to_w: np.ndarray # 4x4 homogeneous transform from gate frame to world frame
    T_q_to_w: np.ndarray # 4x4 homogeneous transform from quadrotor frame to world frame
    quadrotor_pose_world: np.ndarray


class PnPPoseCompose:
    """Estimate quadrotor world pose from a known gate and ``T_g_to_q``."""

    def __init__(self, gate_map: GateMap | None = None) -> None:
        self.gate_map = gate_map if gate_map is not None else GateMap()

    @staticmethod
    def yaw_to_rotation_matrix(yaw_rad: float) -> np.ndarray:
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        return np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @classmethod
    def pose_to_transform(cls, pose: np.ndarray) -> np.ndarray:
        """Build a 4x4 transform from ``[x, y, z, yaw_rad]``."""
        if pose.shape != (4,):
            raise ValueError("Pose must be a numpy array with shape (4,)")

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cls.yaw_to_rotation_matrix(pose[3])
        transform[:3, 3] = pose[:3]
        return transform

    @staticmethod
    def invert_transform(transform: np.ndarray) -> np.ndarray:
        """Invert a rigid 4x4 homogeneous transform."""
        PnPPoseCompose.validate_transform(transform)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        inverse = np.eye(4, dtype=float)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ translation
        return inverse

    @staticmethod
    def transform_to_pose(transform: np.ndarray) -> np.ndarray:
        """Extract ``[x, y, z, yaw_rad]`` from a 4x4 transform."""
        PnPPoseCompose.validate_transform(transform)
        yaw_rad = np.arctan2(transform[1, 0], transform[0, 0])
        return np.array(
            [
                transform[0, 3],
                transform[1, 3],
                transform[2, 3],
                yaw_rad,
            ],
            dtype=float,
        )

    @staticmethod
    def validate_transform(transform: np.ndarray) -> None:
        if not isinstance(transform, np.ndarray):
            raise TypeError("Transform must be a numpy.ndarray")
        if transform.shape != (4, 4):
            raise ValueError("Transform must have shape (4, 4)")

    def get_gate_transform_world(
        self,
        gate_id: str | int,
    ) -> tuple[np.ndarray, np.ndarray]:
        gate_pose_world = self.gate_map.get_gate_pose(gate_id)
        return gate_pose_world, self.pose_to_transform(gate_pose_world)

    def comp_quadrotor_pose(
        self,
        gate_id: str | int,
        T_g_to_q: np.ndarray,
    ) -> PnPPoseComposeResult:
        """Compute ``T_q_to_w = T_g_to_w @ inv(T_g_to_q)``."""
        self.validate_transform(T_g_to_q)
        gate_key = self.gate_map.normalize_gate_id(gate_id)
        gate_pose_world, T_g_to_w = self.get_gate_transform_world(gate_key)
        T_q_to_w = T_g_to_w @ self.invert_transform(T_g_to_q)
        quadrotor_pose_world = self.transform_to_pose(T_q_to_w)
        return PnPPoseComposeResult(
            gate_id=gate_key,
            gate_pose_world=gate_pose_world,
            T_g_to_w=T_g_to_w,
            T_q_to_w=T_q_to_w,
            quadrotor_pose_world=quadrotor_pose_world,
        )

    def get_T_g_to_q(self, gate_pos_quad: np.ndarray) -> np.ndarray:
        """Compute the transform from gate frame to quadrotor frame."""
        if gate_pos_quad.shape != (3,):
            raise ValueError("gate_pos_quad must have shape (3,)")

        T_g_to_q = np.eye(4, dtype=float)
        T_g_to_q[:3, 3] = gate_pos_quad
        return T_g_to_q