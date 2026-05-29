"""Dataset IO: torch-free LeRobot v3 reading and a synthetic dataset builder."""

from __future__ import annotations

from .lerobot_v3 import LeRobotV3Reader, open_dataset
from .materialize import MaterializeResult, materialize
from .synthetic import make_synthetic_v3

__all__ = [
    "LeRobotV3Reader",
    "open_dataset",
    "make_synthetic_v3",
    "materialize",
    "MaterializeResult",
]
