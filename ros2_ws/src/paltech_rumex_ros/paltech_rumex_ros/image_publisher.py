from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class RumexImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__("rumex_image_publisher")
        self.bridge = CvBridge()
        self.declare_parameter("image_dir", str(Path.cwd()))
        self.declare_parameter("frame_rate", 0.5)
        self.declare_parameter("loop_dataset", True)

        image_dir = Path(self.get_parameter("image_dir").get_parameter_value().string_value)
        self.frame_rate = self.get_parameter("frame_rate").get_parameter_value().double_value
        self.loop_dataset = (
            self.get_parameter("loop_dataset").get_parameter_value().bool_value
        )

        self.images = sorted(
            [
                p
                for p in image_dir.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if not self.images:
            self.get_logger().warning(f"No images found in {image_dir}")
        else:
            self.get_logger().info(f"Loaded {len(self.images)} images from {image_dir}")

        self.publisher = self.create_publisher(Image, "rumex/images", 10)
        period = 1.0 / max(self.frame_rate, 1e-3)
        self.timer = self.create_timer(period, self._publish_next)
        self._index = 0

    def _publish_next(self) -> None:
        if not self.images:
            return
        if self._index >= len(self.images):
            if not self.loop_dataset:
                self.get_logger().info("Finished publishing dataset.")
                return
            self._index = 0

        image_path = self.images[self._index]
        image = cv2.imread(str(image_path))
        if image is None:
            self.get_logger().warning(f"Failed to read {image_path}")
            self._index += 1
            return

        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.frame_id = "rumex_camera"
        self.publisher.publish(msg)
        self.get_logger().info(f"Published {image_path.name}")
        self._index += 1


def main(args=None):
    rclpy.init(args=args)
    node = RumexImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
