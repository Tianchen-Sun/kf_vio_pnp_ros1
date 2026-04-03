from pathlib import Path

import numpy as np
from pnp import GateMap, PnPPoseCompose


def test_gate_map_loads_scene_csv() -> None:
    gate_map = GateMap(Path("config/scene.csv"))

    assert "07" in gate_map.gates
    np.testing.assert_allclose(
        gate_map.gates["07"],
        np.array([9.0, 15.0, 1.424, 0.0]),
    )


def test_gate_map_normalizes_multiple_gate_id_formats() -> None:
    gate_map = GateMap(Path("config/scene.csv"))

    np.testing.assert_allclose(
        gate_map.get_gate_pose("gate_05"),
        gate_map.get_gate_pose("05"),
    )
    np.testing.assert_allclose(
        gate_map.get_gate_pose(5),
        gate_map.get_gate_pose("05"),
    )
    assert np.isclose(gate_map.get_gate_pose("05")[3], np.deg2rad(156.8))


def test_comp_quadrotor_pose_with_identity_relative_transform() -> None:
    pnp_pose_composer = PnPPoseCompose(GateMap(Path("config/scene.csv")))
    T_g_to_q = np.eye(4, dtype=float)

    result = pnp_pose_composer.comp_quadrotor_pose("gate_07", T_g_to_q)

    np.testing.assert_allclose(result.T_q_to_w, result.T_g_to_w)
    np.testing.assert_allclose(
        result.quadrotor_pose_world,
        np.array([9.0, 15.0, 1.424, 0.0]),
    )


def test_comp_quadrotor_pose_matches_known_world_pose() -> None:
    pnp_pose_composer = PnPPoseCompose(GateMap(Path("config/scene.csv")))
    expected_quadrotor_pose = np.array([20.0, 10.0, 1.5, np.deg2rad(30.0)])
    T_q_to_w_expected = pnp_pose_composer.pose_to_transform(expected_quadrotor_pose)

    _, T_g_to_w = pnp_pose_composer.get_gate_transform_world("03")
    T_g_to_q = pnp_pose_composer.invert_transform(T_q_to_w_expected) @ T_g_to_w

    result = pnp_pose_composer.comp_quadrotor_pose("03", T_g_to_q)

    np.testing.assert_allclose(result.T_q_to_w, T_q_to_w_expected, atol=1e-9)
    np.testing.assert_allclose(
        result.quadrotor_pose_world,
        expected_quadrotor_pose,
        atol=1e-9,
    )

if __name__ == '__main__':
    
    test_gate_map_loads_scene_csv()
    test_gate_map_normalizes_multiple_gate_id_formats()
    test_comp_quadrotor_pose_with_identity_relative_transform()
    test_comp_quadrotor_pose_matches_known_world_pose()
    print("All tests passed!")
    
    gate_map = GateMap()
    pnp_pose_composer = PnPPoseCompose(gate_map)

    T_g_to_q = np.eye(4, dtype=float)  
    result = pnp_pose_composer.comp_quadrotor_pose(7, T_g_to_q)

    print("gate dict:", gate_map.gates)
    print("gate pose [cx, cy, cz, yaw_rad]:", result.gate_pose_world)
    print("T_g_to_w:\n", result.T_g_to_w)
    print("T_q_to_w:\n", result.T_q_to_w)
    print("quadrotor pose [x, y, z, yaw_rad]:", result.quadrotor_pose_world)
