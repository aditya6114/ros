#!/usr/bin/env python3
"""ROS 2 /cmd_vel-to-USB bridge and encoder odometry for the Beetle base."""
import math
import threading

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

try:
    import serial
except ImportError as exc:
    raise RuntimeError("Install pyserial: sudo apt install python3-serial") from exc


class BeetleSerialBridge(Node):
    def __init__(self):
        super().__init__("beetle_serial_bridge")
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("wheel_diameter_m", 0.106)
        self.declare_parameter("track_width_m", 0.276)
        self.declare_parameter("counts_per_wheel_rev", 5200.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        diameter = float(self.get_parameter("wheel_diameter_m").value)
        self.track = float(self.get_parameter("track_width_m").value)
        cpr = float(self.get_parameter("counts_per_wheel_rev").value)
        self.meters_per_tick = math.pi * diameter / cpr
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_callback, 20)
        self.serial_lock = threading.Lock()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        self.x = self.y = self.yaw = 0.0
        self.last_left = self.last_right = None
        self.last_stamp = None
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.get_logger().info(f"Beetle base connected on {self.port} at {self.baud} baud")

    def _cmd_callback(self, msg):
        # The ESP32 has an independent 300 ms timeout, even if this node fails.
        command = f"CMD {msg.linear.x:.4f} {msg.angular.z:.4f}\n".encode()
        try:
            with self.serial_lock:
                self.ser.write(command)
        except serial.SerialException as exc:
            self.get_logger().error(f"Cannot write motor command: {exc}")

    def _read_loop(self):
        while rclpy.ok():
            try:
                with self.serial_lock:
                    raw = self.ser.readline().decode("ascii", errors="ignore").strip()
            except serial.SerialException as exc:
                self.get_logger().error(f"Encoder serial read failed: {exc}")
                return
            fields = raw.split()
            if len(fields) != 4 or fields[0] != "ENC":
                continue
            try:
                self._update_odometry(int(fields[1]), int(fields[2]))
            except ValueError:
                continue

    def _update_odometry(self, left, right):
        stamp = self.get_clock().now()
        if self.last_left is None:
            self.last_left, self.last_right, self.last_stamp = left, right, stamp
            return
        dt = (stamp - self.last_stamp).nanoseconds / 1e9
        if dt <= 0.0:
            return
        dl = (left - self.last_left) * self.meters_per_tick
        dr = (right - self.last_right) * self.meters_per_tick
        self.last_left, self.last_right, self.last_stamp = left, right, stamp
        distance = (dl + dr) * 0.5
        dtheta = (dr - dl) / self.track
        self.x += distance * math.cos(self.yaw + dtheta * 0.5)
        self.y += distance * math.sin(self.yaw + dtheta * 0.5)
        self.yaw = math.atan2(math.sin(self.yaw + dtheta), math.cos(self.yaw + dtheta))
        qz, qw = math.sin(self.yaw * 0.5), math.cos(self.yaw * 0.5)
        odom = Odometry()
        odom.header.stamp = stamp.to_msg(); odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x; odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz; odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = distance / dt; odom.twist.twist.angular.z = dtheta / dt
        self.odom_pub.publish(odom)
        transform = TransformStamped()
        transform.header = odom.header; transform.child_frame_id = self.base_frame
        transform.transform.translation.x = self.x; transform.transform.translation.y = self.y
        transform.transform.rotation.z = qz; transform.transform.rotation.w = qw
        self.tf_pub.sendTransform(transform)

    def destroy_node(self):
        try:
            with self.serial_lock:
                self.ser.write(b"CMD 0 0\n")
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = BeetleSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
