from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("hw_timing")
@dataclass
class HWTimingConfig(RobotConfig):
    # Motor bus
    port: str = "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = False
    # Explicit path to a calibration JSON file; takes precedence over calibration_dir
    calibration_fpath: Path | None = (
        "/home/portegak/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_L.json"
    )
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
