#!/usr/bin/env python3
"""Detect damaged boxes, estimate their 2D position, and persist an inspection map."""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
import tf2_geometry_msgs
import tf2_ros
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class ObjectMapper(Node):
    def __init__(self):
        super().__init__("object_mapper")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("model_path", "")
        self.declare_parameter(
            "database_path", "~/.ros/warehouse_inspection/detections.json"
        )
        # The trained damage models deliberately contain exactly one class.
        # Keep this configurable for experimentation, but do not accept generic
        # "box" labels from an Open Images fallback model as damaged boxes.
        self.declare_parameter("target_classes", ["damaged_box"])
        self.declare_parameter("camera_hfov_deg", 62.2)
        self.declare_parameter("confidence", 0.55)
        self.declare_parameter("max_range", 4.0)
        self.declare_parameter("duplicate_radius", 0.65)
        self.declare_parameter("confirmation_observations", 5)
        self.declare_parameter("candidate_ttl_sec", 5.0)
        self.declare_parameter("reset_database_on_start", False)
        self.declare_parameter("skip_rate", 3)
        self.declare_parameter("debug_view", False)

        self.image_topic = self.get_parameter("image_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.camera_hfov = math.radians(
            float(self.get_parameter("camera_hfov_deg").value)
        )
        self.confidence = float(self.get_parameter("confidence").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.duplicate_radius = float(self.get_parameter("duplicate_radius").value)
        self.confirmation_observations = max(
            1, int(self.get_parameter("confirmation_observations").value)
        )
        self.candidate_ttl_ns = int(
            max(0.1, float(self.get_parameter("candidate_ttl_sec").value)) * 1e9
        )
        self.reset_database_on_start = bool(
            self.get_parameter("reset_database_on_start").value
        )
        self.skip_rate = max(1, int(self.get_parameter("skip_rate").value))
        self.debug_view = bool(self.get_parameter("debug_view").value)
        self.target_classes = {
            str(name).strip().lower()
            for name in self.get_parameter("target_classes").value
        }
        self.database_path = Path(
            os.path.expanduser(str(self.get_parameter("database_path").value))
        )

        configured_model = os.path.expanduser(
            str(self.get_parameter("model_path").value)
        )
        if not configured_model:
            raise RuntimeError(
                "model_path is required for physical damaged-box detection. "
                "Pass a real-camera damaged_box YOLO .pt file to the launch."
            )
        model_path = configured_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model does not exist: {model_path}")
        self.model = YOLO(model_path)

        transient_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self.lidar_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.objects_pub = self.create_publisher(
            MarkerArray, "/detected_objects_map", transient_qos
        )
        self.objects_data_pub = self.create_publisher(
            String, "/detected_objects_data", transient_qos
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.latest_scan = None
        self.camera_info = None
        self.frame_count = 0
        self.pending_positions = {}
        if self.reset_database_on_start:
            self.object_positions = {}
            self._save_database()
            self.get_logger().info("Started with an empty inspection database")
        else:
            self.object_positions = self._load_database()
        self.create_timer(1.0, self.publish_database)

        self.get_logger().info(
            f"Inspection mapper ready: image={self.image_topic}, "
            f"scan={self.scan_topic}, model={model_path}"
        )

    def _load_database(self):
        if not self.database_path.exists():
            return {}
        try:
            data = json.loads(self.database_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.get_logger().info(
                    f"Loaded inspection memory from {self.database_path}"
                )
                return data
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Could not load inspection memory: {exc}")
        return {}

    def _save_database(self):
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.database_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.object_positions, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.database_path)
        except OSError as exc:
            self.get_logger().error(f"Could not save inspection memory: {exc}")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    @staticmethod
    def _rotate_vector(vector, rotation):
        """Rotate a direction vector by a geometry_msgs quaternion."""
        axis = np.array([rotation.x, rotation.y, rotation.z], dtype=float)
        value = np.asarray(vector, dtype=float)
        scalar = float(rotation.w)
        return (
            2.0 * np.dot(axis, value) * axis
            + (scalar * scalar - np.dot(axis, axis)) * value
            + 2.0 * scalar * np.cross(axis, value)
        )

    def _scan_bearing_for_pixel(self, pixel_x, image_width, image_header):
        """Convert a camera pixel to a lidar bearing using calibrated camera TF."""
        info = self.camera_info
        if info is not None and len(info.k) == 9 and float(info.k[0]) > 0.0:
            fx = float(info.k[0])
            principal_x = float(info.k[2])
            optical_ray = [(pixel_x - principal_x) / fx, 0.0, 1.0]
            camera_frame = info.header.frame_id or image_header.frame_id
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.latest_scan.header.frame_id,
                    camera_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                scan_ray = self._rotate_vector(
                    optical_ray, transform.transform.rotation
                )
                return math.atan2(float(scan_ray[1]), float(scan_ray[0]))
            except Exception as exc:
                self.get_logger().debug(
                    f"Camera-to-lidar calibration TF unavailable; using FOV fallback: {exc}"
                )
        return -(pixel_x / image_width - 0.5) * self.camera_hfov

    def _range_near_angle(self, angle):
        scan = self.latest_scan
        if scan is None or not scan.ranges or scan.angle_increment == 0.0:
            return None

        angle = math.atan2(math.sin(angle), math.cos(angle))
        if angle < scan.angle_min or angle > scan.angle_max:
            if scan.angle_max > math.pi and angle < 0.0:
                angle += 2.0 * math.pi
            else:
                return None

        center = int(round((angle - scan.angle_min) / scan.angle_increment))
        half_window = max(1, int(math.radians(2.0) / abs(scan.angle_increment)))
        values = []
        for index in range(center - half_window, center + half_window + 1):
            if 0 <= index < len(scan.ranges):
                value = float(scan.ranges[index])
                limit = min(float(scan.range_max), self.max_range)
                if math.isfinite(value) and scan.range_min <= value <= limit:
                    values.append(value)
        return float(np.median(values)) if values else None

    def _find_duplicate(self, positions, label, x, y):
        for position in positions.get(label, []):
            if math.hypot(x - position["x"], y - position["y"]) < self.duplicate_radius:
                return position
        return None

    @staticmethod
    def _refine_position(position, x, y, confidence, now):
        observations = int(position.get("observations", 1))
        position["x"] = round(
            (float(position["x"]) * observations + x) / (observations + 1), 3
        )
        position["y"] = round(
            (float(position["y"]) * observations + y) / (observations + 1), 3
        )
        position["confidence"] = round(
            max(float(position.get("confidence", 0.0)), confidence), 3
        )
        position["observations"] = observations + 1
        position["observed_at"] = now

    def _prune_pending(self, now):
        """Discard candidates that were not observed again within the time window."""
        for label in list(self.pending_positions):
            candidates = self.pending_positions[label]
            self.pending_positions[label] = [
                candidate
                for candidate in candidates
                if 0 <= now - int(candidate.get("observed_at", now)) <= self.candidate_ttl_ns
            ]
            if not self.pending_positions[label]:
                del self.pending_positions[label]

    def _store_detection(self, label, x, y, confidence):
        """Refine a confirmed target or promote a repeatedly seen candidate."""
        now = self.get_clock().now().nanoseconds
        self._prune_pending(now)

        existing = self._find_duplicate(self.object_positions, label, x, y)
        if existing is not None:
            self._refine_position(existing, x, y, confidence, now)
            return "updated"

        candidate = self._find_duplicate(self.pending_positions, label, x, y)
        if candidate is None:
            self.pending_positions.setdefault(label, []).append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": 0.0,
                    "confidence": round(confidence, 3),
                    "observations": 1,
                    "observed_at": now,
                    "last_frame": self.frame_count,
                }
            )
            return "pending"

        # Multiple overlapping boxes in one YOLO result must not count as
        # independent temporal confirmations.
        if int(candidate.get("last_frame", -1)) != self.frame_count:
            self._refine_position(candidate, x, y, confidence, now)
            candidate["last_frame"] = self.frame_count

        if int(candidate["observations"]) < self.confirmation_observations:
            return "pending"

        self.pending_positions[label].remove(candidate)
        confirmed = {
            key: value for key, value in candidate.items() if key != "last_frame"
        }
        self.object_positions.setdefault(label, []).append(confirmed)
        return "confirmed"

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.skip_rate or self.latest_scan is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            result = self.model.predict(
                frame, conf=self.confidence, verbose=False
            )[0]
        except Exception as exc:
            self.get_logger().error(f"Image inference failed: {exc}")
            return

        _, image_width = frame.shape[:2]
        for detection in result.boxes:
            class_id = int(detection.cls.item())
            label = str(self.model.names[class_id])
            # Preserve the trained YOLO class spelling in the published data:
            # consumers receive exactly the required `damaged_box` key.
            normalized_label = label.strip().lower()
            if self.target_classes and normalized_label not in self.target_classes:
                continue

            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().tolist()
            confidence = float(detection.conf.item())
            center_x = (x1 + x2) * 0.5
            bearing = self._scan_bearing_for_pixel(center_x, image_width, msg.header)
            distance = self._range_near_angle(bearing)
            if distance is None:
                continue

            point = PointStamped()
            point.header = self.latest_scan.header
            point.point.x = distance * math.cos(bearing)
            point.point.y = distance * math.sin(bearing)

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    point.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                mapped = tf2_geometry_msgs.do_transform_point(point, transform)
            except Exception as exc:
                self.get_logger().warning(f"Map transform unavailable: {exc}")
                continue

            x = float(mapped.point.x)
            y = float(mapped.point.y)
            status = self._store_detection(normalized_label, x, y, confidence)
            if status != "pending":
                self._save_database()
                self.publish_database()
            if status == "confirmed":
                self.get_logger().info(
                    f"Confirmed {normalized_label} at map coordinate ({x:.2f}, {y:.2f})"
                )

            if self.debug_view:
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (int(x1), max(18, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

        if self.debug_view:
            cv2.imshow("Warehouse damage inspection", frame)
            cv2.waitKey(1)

    def publish_database(self):
        self._prune_pending(self.get_clock().now().nanoseconds)

        data = String()
        data.data = json.dumps(self.object_positions, sort_keys=True)
        self.objects_data_pub.publish(data)

        markers = MarkerArray()
        # RViz retains markers by namespace and ID until explicitly deleted.
        # Clearing first prevents labels from an older/larger database lingering.
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        marker_id = 0
        for label, positions in sorted(self.object_positions.items()):
            for position in positions:
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "warehouse_damage"
                marker.id = marker_id
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                marker.pose.position.x = float(position["x"])
                marker.pose.position.y = float(position["y"])
                marker.pose.position.z = 0.7
                marker.pose.orientation.w = 1.0
                marker.scale.z = 0.32
                marker.color.r = 1.0
                marker.color.g = 0.1
                marker.color.b = 0.05
                marker.color.a = 1.0
                marker.text = label
                marker.frame_locked = True
                markers.markers.append(marker)
                marker_id += 1
        self.objects_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
