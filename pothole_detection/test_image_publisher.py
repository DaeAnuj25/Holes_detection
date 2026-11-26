"""Independent camera stand-in that publishes local media as ROS Image frames.

It intentionally knows nothing about the detector implementation: point the
existing detector at the same image topic, then replace this node with a real
camera driver later without changing the inference node or dashboard.
"""
import math
from pathlib import Path
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('source', '')
        self.declare_parameter('webcam_index', -1)
        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('frame_id', 'test_camera')

        self.topic = str(self.get_parameter('topic').value)
        source = str(self.get_parameter('source').value).strip()
        webcam_index = int(self.get_parameter('webcam_index').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        if not self.topic.startswith('/'):
            raise ValueError('topic must be an absolute ROS topic (for example /camera/image_raw)')
        if not math.isfinite(self.publish_rate) or self.publish_rate <= 0:
            raise ValueError('publish_rate must be a positive finite number')
        if source and webcam_index >= 0:
            raise ValueError('Set either source or webcam_index, not both')
        if not source and webcam_index < 0:
            raise ValueError('Set source to an image/video/folder path or set webcam_index (for example 0)')

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, self.topic, qos_profile_sensor_data)
        self.capture = None
        self.image_paths = []
        self.image_index = 0
        self.single_image = None
        self.source_kind = None
        self.video_path = None
        self.frame_count = 0
        self.last_read_warning = 0.0
        self._configure_source(source, webcam_index)
        self.timer = self.create_timer(1.0 / self.publish_rate, self._publish_frame)
        self.get_logger().info(
            'Publishing %s frames to %s at %.2f FPS (sensor-data QoS)' %
            (self.source_kind, self.topic, self.publish_rate))

    def _configure_source(self, source, webcam_index):
        if webcam_index >= 0:
            self.capture = cv2.VideoCapture(webcam_index)
            self.source_kind = 'webcam %d' % webcam_index
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
                raise RuntimeError('Could not open webcam index %d' % webcam_index)
            return

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError('Media source does not exist: %s' % path)
        if path.is_dir():
            self.image_paths = sorted(
                (item for item in path.iterdir()
                 if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda item: item.name.lower(),
            )
            if not self.image_paths:
                raise ValueError('No supported image files found in folder: %s' % path)
            self.source_kind = 'image folder %s (%d images)' % (path, len(self.image_paths))
            return

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            self.single_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if self.single_image is None:
                raise ValueError('Could not decode image file: %s' % path)
            self.source_kind = 'image %s (repeated)' % path
            return

        self.video_path = str(path)
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError('Could not open video file: %s' % path)
        self.source_kind = 'looping video %s' % path

    def _next_frame(self):
        if self.single_image is not None:
            return self.single_image
        if self.image_paths:
            path = self.image_paths[self.image_index]
            self.image_index = (self.image_index + 1) % len(self.image_paths)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning('Skipping unreadable image: %s' % path)
                return None
            return image

        ok, frame = self.capture.read()
        if ok and frame is not None:
            return frame
        if self.source_kind.startswith('looping video'):
            # Seeking is preferred; reopening is a fallback for codecs that do
            # not support reliable frame-position reset after EOF.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.capture.release()
                self.capture = cv2.VideoCapture(self.video_path)
                ok, frame = self.capture.read() if self.capture.isOpened() else (False, None)
            if ok and frame is not None:
                return frame
            self.get_logger().error('Unable to read or loop the video; stopping publisher timer.')
            self.timer.cancel()
            return None

        now = time.monotonic()
        if now - self.last_read_warning >= 5.0:
            self.get_logger().warning('No frame received from webcam; check camera permissions/connection.')
            self.last_read_warning = now
        return None

    def _publish_frame(self):
        frame = self._next_frame()
        if frame is None:
            return
        try:
            message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.frame_id
            self.publisher.publish(message)
            self.frame_count += 1
            if self.frame_count == 1 or self.frame_count % max(1, round(self.publish_rate * 10)) == 0:
                self.get_logger().info('Published %d frames on %s' % (self.frame_count, self.topic))
        except Exception as exc:
            self.get_logger().error('Could not publish image frame: %s' % exc)

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TestImagePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt,):
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print('test_image_publisher: %s' % exc)
        raise SystemExit(2)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
