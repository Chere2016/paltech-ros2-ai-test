from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Ensure the shared paltech_rumex package is importable when running inside the ROS workspace.
_FILE = Path(__file__).resolve()
env_root = os.environ.get("PALTECH_RUMEX_ROOT")
if env_root and env_root not in sys.path:
    sys.path.append(env_root)

if "paltech_rumex" not in sys.modules:
    for parent in _FILE.parents:
        candidate = parent / "paltech_rumex"
        if candidate.exists() and str(parent) not in sys.path:
            sys.path.append(str(parent))
            break

from paltech_rumex import RumexDetector  # noqa: E402


class RumexDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("rumex_detection")
        self.bridge = CvBridge()

        self.declare_parameter("model_path", str(Path.cwd() / "data/yolo11m_finetuned.pt"))
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("cluster_distance", 0.0)
        self.declare_parameter("cluster_ratio", 0.12)
        self.declare_parameter("output_dir", "")

        model_path = Path(
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        cluster_distance = (
            self.get_parameter("cluster_distance").get_parameter_value().double_value
        )
        cluster_ratio = self.get_parameter("cluster_ratio").get_parameter_value().double_value
        conf = self.get_parameter("conf").get_parameter_value().double_value
        output_dir_value = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self.output_dir: Optional[Path] = Path(output_dir_value) if output_dir_value else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        cluster_distance_px = cluster_distance if cluster_distance > 0 else None
        self.detector = RumexDetector(
            model_path=model_path,
            conf=conf,
            cluster_distance_px=cluster_distance_px,
            cluster_distance_ratio=cluster_ratio,
        )

        self.subscription = self.create_subscription(
            Image, "rumex/images", self._image_callback, 10
        )
        self.publisher = self.create_publisher(Float32MultiArray, "rumex/plant_centers", 10)
        self.get_logger().info(f"Rumex detection node ready (model={model_path})")

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        image_name = f"ros_frame_{msg.header.stamp.sec}_{msg.header.stamp.nanosec}.png"
        result = self.detector.run_on_array(
            image,
            image_name=image_name,
            output_dir=self.output_dir,
        )

        centers = [coord for center in result.plant_centers for coord in center]
        msg_out = Float32MultiArray()
        msg_out.data = [float(value) for value in centers]
        self.publisher.publish(msg_out)
        self.get_logger().info(
            f"Detected {len(result.plants)} plants in {result.runtime_sec:.3f}s"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RumexDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
