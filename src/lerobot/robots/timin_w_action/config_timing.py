from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("timing")
@dataclass
class TimingConfig(RobotConfig):
    port: str = "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True
    calibration_fpath: Path | None = None
    # cameras
    id: str | None = None
    cameras: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: {
            "base_0_rgb": RealSenseCameraConfig(
                serial_number_or_name="218622273888",
                fps=30,
                width=640,
                height=480,
            ),
            "left_wrist_0_rgb": RealSenseCameraConfig(
                serial_number_or_name="821212061298",
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist_0_rgb": RealSenseCameraConfig(
                serial_number_or_name="821212062747",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
