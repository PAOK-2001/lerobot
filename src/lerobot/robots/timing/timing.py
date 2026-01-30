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

import logging
import time
from functools import cached_property
from threading import Event, Lock, Thread

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot

logger = logging.getLogger(__name__)


class Timing(Robot):
    name = "timing"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
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
        self.cameras = make_cameras_from_configs(config.cameras)
        self.move_lock = Lock()
        self.last_timestep = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

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
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

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
        pass

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        start = time.perf_counter()

        # MOTOR READ, not blocking 
        time.sleep(0.001)

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
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        # Send goal position to the arm

        # check if first action in chunk
        timestep = action["joint_2.pos"]
        if timestep != self.last_timestep:
            camera_1_ts = action["joint_2.pos"]
            camera_2_ts = action["joint_3.pos"]
            camera_3_ts = action["joint_4.pos"]
            policy_start = action["joint_5.pos"]
            policy_end = action["joint_6.pos"]
            action_received = time.perf_counter()

            logger.info(
                "TIMESTAMPS:\n"
                f"Cameras: {camera_1_ts}, {camera_2_ts}, {camera_3_ts}\n"
                f"Policy start: {policy_start}\n"
                f"Policy end: {policy_end}\n"
                f"Action received: {action_received}\n"
                f"Total end-to-end: {action_received - camera_1_ts}"
            )

            self.last_timestep = timestep

        # MOTOR WRITE, not blocking
        time.sleep(0.001)

    @check_if_not_connected
    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
