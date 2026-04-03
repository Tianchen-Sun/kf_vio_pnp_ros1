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
