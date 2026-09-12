#!/usr/bin/env python3
"""Navigate to a named object stored by the inspection mapper."""

import json
import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener


class GoToObject(Node):
    def __init__(self):
        super().__init__("go_to_object_node")
        self.declare_parameter("target_topic", "/inspection/target")
        self.declare_parameter("objects_topic", "/detected_objects_data")
        self.declare_parameter("exploration_complete_topic", "/inspection/exploration_complete")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("stand_off_distance", 0.65)
        self.declare_parameter("require_exploration_complete", True)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.stand_off = float(self.get_parameter("stand_off_distance").value)
        self.require_complete = bool(self.get_parameter("require_exploration_complete").value)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, str(self.get_parameter("objects_topic").value), self._objects_callback, qos)
        self.create_subscription(Bool, str(self.get_parameter("exploration_complete_topic").value), self._complete_callback, qos)
        self.create_subscription(String, str(self.get_parameter("target_topic").value), self._target_callback, 10)
        self.status_pub = self.create_publisher(String, "/inspection/navigation_status", qos)
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.objects_data = {}
        self.exploration_complete = not self.require_complete
        self.goal_active = False
        self._publish_status("ready")
        self.get_logger().info(f"Object navigation ready; publish a label on {self.get_parameter('target_topic').value}.")

    @staticmethod
    def _normalize(value):
        return str(value).strip().lower().replace("_", " ")

    def _publish_status(self, value):
        msg = String()
        msg.data = value
        self.status_pub.publish(msg)

    def _objects_callback(self, msg):
        try:
            parsed = json.loads(msg.data)
            if isinstance(parsed, dict):
                self.objects_data = parsed
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid object database: {exc}")

    def _complete_callback(self, msg):
        self.exploration_complete = bool(msg.data)

    def _robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(), timeout=Duration(seconds=0.3)
            )
        except TransformException as exc:
            self.get_logger().warning(f"Robot pose is unavailable: {exc}")
            return None
        return (float(transform.transform.translation.x), float(transform.transform.translation.y))

    def _target_callback(self, msg):
        requested = self._normalize(msg.data)
        if not requested:
            return
        if self.goal_active:
            self.get_logger().warning("A target navigation goal is already active.")
            self._publish_status("busy")
            return
        if self.require_complete and not self.exploration_complete:
            self.get_logger().warning("Exploration is not complete. The target command was not executed.")
            self._publish_status("waiting_for_exploration")
            return
        if not self.nav_client.server_is_ready():
            self.get_logger().warning("NavigateToPose action server is not ready.")
            self._publish_status("waiting_for_nav2")
            return
        matches = []
        for label, positions in self.objects_data.items():
            if self._normalize(label) == requested and isinstance(positions, list):
                matches.extend(position for position in positions if isinstance(position, dict))
        if not matches:
            available = ", ".join(sorted(self.objects_data)) or "none"
            self.get_logger().warning(f"No stored '{requested}' object. Available labels: {available}")
            self._publish_status("target_not_found")
            return
        robot = self._robot_position()
        if robot is None:
            self._publish_status("waiting_for_tf")
            return
        try:
            target = min(matches, key=lambda item: math.hypot(float(item["x"]) - robot[0], float(item["y"]) - robot[1]))
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().warning(f"Stored target coordinates are invalid: {exc}")
            self._publish_status("invalid_target")
            return
        self._send_goal(float(target["x"]), float(target["y"]), robot, requested)

    def _send_goal(self, object_x, object_y, robot, label):
        dx, dy = object_x - robot[0], object_y - robot[1]
        distance = math.hypot(dx, dy)
        if distance > 1e-6:
            goal_x = object_x - self.stand_off * dx / distance
            goal_y = object_y - self.stand_off * dy / distance
            yaw = math.atan2(dy, dx)
        else:
            goal_x, goal_y, yaw = robot[0], robot[1], 0.0
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(yaw * 0.5)
        self.goal_active = True
        self._publish_status(f"navigating:{label}")
        self.get_logger().info(f"Navigating to '{label}' via stand-off pose ({goal_x:.2f}, {goal_y:.2f}).")
        self.nav_client.send_goal_async(goal).add_done_callback(self._goal_response)

    def _goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Could not send object goal: {exc}")
            self.goal_active = False
            self._publish_status("failed_to_send")
            return
        if not handle.accepted:
            self.get_logger().warning("Object navigation goal was rejected.")
            self.goal_active = False
            self._publish_status("rejected")
            return
        handle.get_result_async().add_done_callback(self._goal_result)

    def _goal_result(self, future):
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().error(f"Object navigation failed: {exc}")
            status = GoalStatus.STATUS_UNKNOWN
        self.goal_active = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Reached the requested inspection location.")
            self._publish_status("arrived")
        else:
            self.get_logger().warning(f"Object navigation ended with status {status}.")
            self._publish_status(f"failed:{status}")


def main(args=None):
    rclpy.init(args=args)
    node = GoToObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
