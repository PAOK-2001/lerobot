# run_timing_inference.py
import threading

# Import YOUR robot directly
from lerobot.robots.timing import Timing, TimingConfig

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.realsense import RealSenseCameraConfig

import logging

logging.basicConfig(format="%(pathname)s:%(lineno)d - %(message)s")


# # 1. Camera config
cam_config = {
    "wrist": RealSenseCameraConfig(
        serial_number_or_name="218622273888",
        fps=30,
        width=640,
        height=480,
    ),
    "left": RealSenseCameraConfig(
        serial_number_or_name="821212061298",
        fps=30,
        width=640,
        height=480,
    ),
    "right": RealSenseCameraConfig(
        serial_number_or_name="821212062747",
        fps=30,
        width=640,
        height=480,
    ),
}

# 2. YOUR robot config (not parsed by draccus)
robot_cfg = TimingConfig(
    cameras=cam_config,
    # calibration_fpath="/home/portegak/.cache/huggingface/lerobot/calibration/robots/timing/timing.json",
)

# 3. Client config - pass robot_cfg directly
client_cfg = RobotClientConfig(
    robot=robot_cfg,  # <-- Your custom config works here!
    server_address="127.0.0.1:8080",
    policy_type="pi05",
    pretrained_name_or_path="lerobot/pi05_base",
    policy_device="cuda",
    actions_per_chunk=50,
    chunk_size_threshold=0.5,
)

# 4. Run
client = RobotClient(client_cfg)

if client.start():
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task="Your task instruction")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        action_receiver_thread.join()
        visualize_action_queue_size(client.action_queue_size)
