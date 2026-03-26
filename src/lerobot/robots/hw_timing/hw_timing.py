#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
from functools import cached_property
from pathlib import Path
from threading import Lock

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_hw_timing import HWTimingConfig

logger = logging.getLogger(__name__)

SAMPLES = 510

# Translation layer: robot feature-space joint names → physical bus motor names (SO follower)
JOINT_TO_MOTOR = {
    "joint_1": "shoulder_pan",
    "joint_2": "shoulder_lift",
    "joint_3": "elbow_flex",
    "joint_4": "wrist_flex",
    "joint_5": "wrist_roll",
    "joint_6": "gripper",
}


class HWTiming(Robot):
    name = "hw_timing"
    config_class = HWTimingConfig

    def __init__(self, config: HWTimingConfig):
        super().__init__(config)
        self.config = config

        if config.calibration_fpath is not None:
            self.calibration_fpath = Path(config.calibration_fpath)
            self.calibration = {}
            if self.calibration_fpath.is_file():
                self._load_calibration()

        # Full 8-motor feature set (joint_7 and gripper are virtual — no physical write)
        self.fake_motors = {
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper",
        }

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode),
                "shoulder_lift": Motor(2, "sts3215", norm_mode),
                "elbow_flex": Motor(3, "sts3215", norm_mode),
                "wrist_flex": Motor(4, "sts3215", norm_mode),
                "wrist_roll": Motor(5, "sts3215", norm_mode),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        self.move_lock = Lock()
        self.last_timestep = None
        self.timings = []

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Use fake_motors to include all 8 joints in the feature set
        return {f"{motor}.pos": float for motor in self.fake_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        # Real motor read — latency is measured, values are discarded
        self.bus.sync_read("Present_Position")

        obs_dict = {motor: 0.0 for motor in self._motors_ft}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        times = []

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key], t = cam.async_read()
            times.append(t)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        obs_dict["joint_1.pos"] = times[0]
        obs_dict["joint_2.pos"] = times[1]
        obs_dict["joint_3.pos"] = times[2]

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send zeros to motors; extract timing info from action channels.

        Actions are used to carry timing timestamps, not real motor targets.
        joint_1..joint_6 are written as zeros to the physical bus.
        joint_7 and gripper are virtual and receive no physical write.

        Returns:
            RobotAction: the action actually sent (all zeros for physical motors).
        """
        # Send zeros to all physical motors (joint_1..joint_6)
        zeros = {motor: 0.0 for motor in self.bus.motors}
        self.bus.sync_write("Goal_Position", zeros)

        # Check if first action in chunk (deduplicate by camera_1 timestamp)
        camera_1_ts = action["joint_2.pos"]
        if camera_1_ts != self.last_timestep:
            camera_2_ts = action["joint_3.pos"]
            camera_3_ts = action["joint_4.pos"]
            policy_start = action["joint_5.pos"]
            policy_end = action["joint_6.pos"]
            action_received = time.perf_counter()

            obs_fetching = policy_start - camera_1_ts
            policy_latency = policy_end - policy_start
            obs_to_action = action_received - camera_1_ts
            latency = {
                "obs_fetching": obs_fetching,
                "policy_latency": policy_latency,
                "obs_to_action": obs_to_action,
            }

            logger.info(f"Collected {len(self.timings)}/{SAMPLES} samples.")
            if len(self.timings) < SAMPLES:
                self.timings.append(latency)

            if len(self.timings) == SAMPLES:
                logger.info(f"!!!!!!!!!!! LAST {obs_to_action * 1000:.2f} ms. SAFE TO STOP !!!!!!!!")
                with open("latency_timings.json", "w") as f:
                    json.dump(self.timings, f, indent=4)

            self.last_timestep = camera_1_ts

        return {f"{motor}.pos": 0.0 for motor in self.bus.motors}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        with open("latency_timings.json", "w") as f:
            json.dump(self.timings, f, indent=4)

        logger.info(f"{self} disconnected.")
