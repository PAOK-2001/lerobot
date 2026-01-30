from dataclasses import dataclass, field
from typing import TypeAlias

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class TimingConfig:
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
