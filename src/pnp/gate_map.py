from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class GateMap:
    """Load gate poses from ``scene.csv`` by two-digit gate id."""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "config" / "scene.csv"
        self.csv_path = Path(csv_path)
        self.gates = self._load_gates()

    @staticmethod
    def normalize_gate_id(gate_id: str | int) -> str:
        """Normalize ``gate_07`` / ``07`` / ``7`` into ``07``."""
        if isinstance(gate_id, int):
            return f"{gate_id:02d}"

        text = str(gate_id).strip()
        if text.startswith("gate_"):
            text = text[-2:]

        if text.isdigit():
            return f"{int(text):02d}"

        raise ValueError(f"Unsupported gate id format: {gate_id}")

    def _load_gates(self) -> dict[str, np.ndarray]:
        gates: dict[str, np.ndarray] = {}

        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row["row_type"] != "gate":
                    continue

                gate_key = self.normalize_gate_id(row["name"])
                yaw_rad = np.deg2rad(float(row["yaw_deg"]))
                gates[gate_key] = np.array(
                    [
                        float(row["cx"]),
                        float(row["cy"]),
                        float(row["cz"]),
                        yaw_rad,
                    ],
                    dtype=float,
                )

        if not gates:
            raise ValueError(f"No gate rows found in {self.csv_path}")

        return gates

    def get_gate_pose(self, gate_id: str | int) -> np.ndarray:
        """Return gate pose in world frame as ``[cx, cy, cz, yaw_rad]``."""
        gate_key = self.normalize_gate_id(gate_id)
        if gate_key not in self.gates:
            raise KeyError(f"Gate id {gate_id} not found in {self.csv_path}")
        return self.gates[gate_key].copy()


from dataclasses import dataclass


@dataclass(frozen=True)
class GateDetection:
    """Decoded gate detection from a one-hot PoseArray message."""
    gate_id: int            # logical gate index (0, 1, 2, ...)
    position: np.ndarray    # [x, y, z] in quadrotor frame (x already sign-corrected)
    is_front: bool          # True = drone approaching gate, False = flying away


class GatePoseArrayDecoder:
    """
    Decode a one-hot PoseArray into a GateDetection.

    PoseArray index layout:
      even index 2k   -> gate k, front side (drone approaching, x > 0 as received)
      odd  index 2k+1 -> gate k, back  side (drone departing,  x sign is flipped)

    Only the single non-zero element is considered valid.
    """

    @staticmethod
    def decode(poses) -> GateDetection | None:
        """
        Find the valid detection in a sequence of geometry_msgs/Pose objects.

        Args:
            poses: iterable of geometry_msgs/Pose (from PoseArray.poses)

        Returns:
            GateDetection if a non-zero entry is found, else None.
        """
        for array_idx, pose in enumerate(poses):
            x, y, z = pose.position.x, pose.position.y, pose.position.z

            if x == 0.0 and y == 0.0 and z == 0.0:
                continue  # empty slot

            gate_id = array_idx // 2
            is_front = (array_idx % 2 == 0)


            return GateDetection(
                gate_id=gate_id,
                position=np.array([x, y, z], dtype=float),
                is_front=is_front,
            )

        return None  # all entries were zero
