"""ROS 2 image-to-pothole-detections node with an embedded local dashboard."""
import json
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .distance import DistanceEstimator


class DashboardState:
    """Small thread-safe snapshot shared between ROS callbacks and HTTP UI."""
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.data = {
            'running': True, 'input_topic': '', 'output_topic': '',
            'model': '', 'fps_limit': 5.0, 'processed_frames': 0,
            'skipped_frames': 0, 'last_inference_ms': None,
            'last_error': None, 'detections': [], 'last_result': None,
            'topics': [], 'preview': None,
        }

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def snapshot(self):
        with self.lock:
            result = dict(self.data)
            result.pop('preview', None)
            result['uptime_s'] = round(time.monotonic() - self.started, 1)
            try:
                result['system_load_1m'] = round(os.getloadavg()[0], 2)
            except (AttributeError, OSError):
                result['system_load_1m'] = None
            try:
                pages = int(Path('/proc/self/statm').read_text().split()[1])
                result['process_rss_mb'] = round(pages * os.sysconf('SC_PAGE_SIZE') / 1048576, 1)
            except (OSError, ValueError, IndexError):
                result['process_rss_mb'] = None
            return result

    def preview(self):
        with self.lock:
            return self.data.get('preview')


class DashboardHandler(BaseHTTPRequestHandler):
    state = None
    html_path = None
    config_queue = None

    def log_message(self, fmt, *args):
        # Keep HTTP polling from flooding the ROS terminal.
        return

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            try:
                self._send(200, Path(self.html_path).read_bytes(), 'text/html; charset=utf-8')
            except OSError:
                self._send(500, b'Dashboard web files are missing.', 'text/plain')
        elif path == '/api/status':
            self._send(200, json.dumps(self.state.snapshot()).encode(), 'application/json')
        elif path == '/api/preview.jpg':
            image = self.state.preview()
            if image:
                self._send(200, image, 'image/jpeg')
            else:
                self._send(404, b'No camera frame yet', 'text/plain')
        else:
            self._send(404, b'Not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/config':
            self._send(404, b'Not found', 'text/plain')
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size > 8192 or size <= 0:
                raise ValueError('Request body must be between 1 and 8192 bytes')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or not isinstance(body.get('input_topic'), str) or not isinstance(body.get('output_topic'), str):
                raise ValueError('Provide input_topic and output_topic as strings')
            input_topic = body['input_topic'].strip()
            output_topic = body['output_topic'].strip()
            topic_pattern = r'/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*'
            if not re.fullmatch(topic_pattern, input_topic) or not re.fullmatch(topic_pattern, output_topic):
                raise ValueError('Use valid absolute ROS topic names, e.g. /camera/image_raw')
            self.config_queue.put({'input_topic': input_topic, 'output_topic': output_topic})
            self._send(202, json.dumps({'accepted': True}).encode(), 'application/json')
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({'error': str(exc)}).encode(), 'application/json')


class PotholeDetector(Node):
    def __init__(self):
        super().__init__('pothole_detector')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detections_topic', '/pothole/detections')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('dashboard_host', '127.0.0.1')
        self.declare_parameter('dashboard_port', 8080)

        self.state = DashboardState()
        self.config_queue = queue.Queue()
        self.bridge = CvBridge()
        self.estimator = DistanceEstimator()
        self.last_inference_time = 0.0
        self.model = self._load_model()
        self.model_path = self._model_path()
        self.publisher = None
        self.subscription = None
        self._set_topics(
            self.get_parameter('image_topic').value,
            self.get_parameter('detections_topic').value,
        )
        self.state.update(model=self.model_path, fps_limit=5.0)
        self.config_timer = self.create_timer(0.25, self._apply_pending_config)
        self.topics_timer = self.create_timer(1.0, self._refresh_topic_list)
        self._start_dashboard()
        self.get_logger().info('Pothole detector ready. Input=%s Output=%s' % (
            self.input_topic, self.output_topic))

    def _model_path(self):
        configured = self.get_parameter('model_path').value
        if configured:
            return str(Path(configured).expanduser().resolve())
        repo_model = Path(__file__).resolve().parents[1] / 'new_model.pt'
        if repo_model.is_file():
            return str(repo_model)
        return str(Path(get_package_share_directory('pothole_detection')) / 'new_model.pt')

    def _load_model(self):
        path = self._model_path()
        if not Path(path).is_file():
            raise FileNotFoundError('Model not found: %s (set model_path)' % path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Ultralytics is missing; install requirements.txt') from exc
        self.get_logger().info('Loading model: %s' % path)
        return YOLO(path)

    def _set_topics(self, input_topic, output_topic):
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
        if self.publisher is not None:
            self.destroy_publisher(self.publisher)
        self.input_topic, self.output_topic = input_topic, output_topic
        self.publisher = self.create_publisher(String, output_topic, 10)
        self.subscription = self.create_subscription(
            Image, input_topic, self._image_callback, qos_profile_sensor_data)
        self.state.update(input_topic=input_topic, output_topic=output_topic,
                          last_error=None)
        self.get_logger().info('Topics set: Image %s -> JSON String %s' %
                               (input_topic, output_topic))

    def _apply_pending_config(self):
        latest = None
        while True:
            try:
                latest = self.config_queue.get_nowait()
            except queue.Empty:
                break
        if latest and (latest['input_topic'] != self.input_topic or
                       latest['output_topic'] != self.output_topic):
            self._set_topics(latest['input_topic'], latest['output_topic'])

    def _refresh_topic_list(self):
        topics = sorted(name for name, types in self.get_topic_names_and_types()
                        if 'sensor_msgs/msg/Image' in types)
        self.state.update(topics=topics)

    def _image_callback(self, msg):
        now = time.monotonic()
        if now - self.last_inference_time < 0.2:
            data = self.state.snapshot()
            self.state.update(skipped_frames=data['skipped_frames'] + 1)
            return
        self.last_inference_time = now
        started = time.perf_counter()
        try:
            # Decode/convert directly to mono8 before inference. Replicating the
            # grayscale plane makes a 3-channel tensor accepted by YOLO models.
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if gray.ndim != 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            model_input = np.repeat(gray[:, :, None], 3, axis=2)
            confidence = float(self.get_parameter('confidence_threshold').value)
            results = self.model.predict(source=model_input, conf=confidence,
                                         verbose=False)
            result = results[0]
            detections = self._parse_detections(result, gray.shape)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            payload = {
                'stamp': datetime.now(timezone.utc).isoformat(),
                'frame_id': msg.header.frame_id,
                'image': {'width': int(gray.shape[1]), 'height': int(gray.shape[0]),
                          'encoding_used_for_inference': 'grayscale replicated to 3 channels'},
                'processing_ms': round(elapsed_ms, 2),
                'distance_note': 'Distances are uncalibrated placeholders (null).',
                'detections': detections,
            }
            out = String()
            out.data = json.dumps(payload, separators=(',', ':'))
            self.publisher.publish(out)
            preview = self._make_preview(gray, detections)
            ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self.state.lock:
                self.state.data.update(
                    processed_frames=self.state.data['processed_frames'] + 1,
                    last_inference_ms=round(elapsed_ms, 2), detections=detections,
                    last_result=payload,
                    preview=encoded.tobytes() if ok else None,
                    last_error=None,
                )
        except Exception as exc:  # Keep the node alive for transient bad frames.
            self.state.update(last_error=str(exc))
            self.get_logger().error('Frame processing failed: %s' % exc)

    def _parse_detections(self, result, image_shape):
        detections = []
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return detections
        names = getattr(result, 'names', {}) or getattr(self.model, 'names', {}) or {}
        for i, coords in enumerate(boxes.xyxy.cpu().numpy()):
            class_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            bbox = [round(float(value), 2) for value in coords]
            estimate = self.estimator.estimate(bbox, image_shape)
            detections.append({
                'class_id': class_id,
                'class_name': str(names.get(class_id, class_id)) if hasattr(names, 'get') else str(class_id),
                'confidence': round(score, 4),
                'bbox_xyxy_px': bbox,
                'center_px': [round((bbox[0] + bbox[2]) / 2, 2),
                              round((bbox[1] + bbox[3]) / 2, 2)],
                'distance': estimate.as_dict(),
            })
        return detections

    @staticmethod
    def _make_preview(gray, detections):
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox_xyxy_px'])
            cv2.rectangle(view, (x1, y1), (x2, y2), (40, 220, 120), 2)
            label = '%s %.0f%%' % (detection['class_name'], detection['confidence'] * 100)
            cv2.putText(view, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 120), 2)
        return view

    def _start_dashboard(self):
        handler = type('BoundDashboardHandler', (DashboardHandler,), {})
        handler.state = self.state
        handler.config_queue = self.config_queue
        try:
            share = Path(get_package_share_directory('pothole_detection'))
        except Exception:
            share = Path(__file__).resolve().parents[1]
        handler.html_path = str(share / 'web' / 'index.html')
        host = str(self.get_parameter('dashboard_host').value)
        port = int(self.get_parameter('dashboard_port').value)
        self.http_server = ThreadingHTTPServer((host, port), handler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever,
                                            name='pothole-dashboard', daemon=True)
        self.http_thread.start()
        self.get_logger().info('Dashboard: http://%s:%d' % (host, port))

    def destroy_node(self):
        if hasattr(self, 'http_server'):
            self.http_server.shutdown()
            self.http_server.server_close()
        self.state.update(running=False)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PotholeDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
