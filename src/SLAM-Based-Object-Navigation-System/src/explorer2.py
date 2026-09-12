#!/usr/bin/env python3
"""Explore an occupancy grid by sending clustered frontier goals to Nav2."""

import math
from collections import deque

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("min_goal_distance", 1.0)
        self.declare_parameter("skip_radius", 0.75)
        self.declare_parameter("frontier_radius_check", 0.8)
        self.declare_parameter("min_unexplored_fraction", 0.20)
        self.declare_parameter("min_frontier_cluster_size", 5)
        self.declare_parameter("empty_map_updates_before_complete", 3)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.skip_radius = float(self.get_parameter("skip_radius").value)
        self.frontier_radius = float(self.get_parameter("frontier_radius_check").value)
        self.min_unexplored = float(self.get_parameter("min_unexplored_fraction").value)
        self.min_cluster_size = int(self.get_parameter("min_frontier_cluster_size").value)
        self.empty_updates_required = int(
            self.get_parameter("empty_map_updates_before_complete").value
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            self._map_callback,
            map_qos,
        )
        self.complete_pub = self.create_publisher(Bool, "/inspection/exploration_complete", state_qos)
        self.state_pub = self.create_publisher(String, "/inspection/state", state_qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.map_data = None
        self.map_info = None
        self.goal_active = False
        self.complete = False
        self.current_goal = None
        self.blocked_points = []
        self.map_version = 0
        self.last_empty_map_version = -1
        self.empty_map_updates = 0
        self.create_timer(1.0, self._try_next_goal)
        self._publish_state("waiting_for_map")

    def _publish_state(self, value):
        msg = String()
        msg.data = value
        self.state_pub.publish(msg)

    def _publish_complete(self, value):
        msg = Bool()
        msg.data = value
        self.complete_pub.publish(msg)

    def _map_callback(self, msg):
        self.map_data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.map_version += 1

    def _robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2)
            )
        except TransformException as exc:
            self.get_logger().debug(f"Robot pose is not ready: {exc}")
            return None
        return (float(transform.transform.translation.x), float(transform.transform.translation.y))

    def _cell_to_world(self, cell):
        row, column = cell
        return (
            self.map_info.origin.position.x + (column + 0.5) * self.map_info.resolution,
            self.map_info.origin.position.y + (row + 0.5) * self.map_info.resolution,
        )

    def _find_frontier_clusters(self):
        free = self.map_data == 0
        unknown = self.map_data == -1
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        frontier = free & adjacent_unknown
        visited = np.zeros_like(frontier, dtype=bool)
        clusters = []
        height, width = frontier.shape
        for row, column in np.argwhere(frontier):
            if visited[row, column]:
                continue
            queue = deque([(int(row), int(column))])
            visited[row, column] = True
            cluster = []
            while queue:
                current = queue.popleft()
                cluster.append(current)
                cr, cc = current
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < height and 0 <= nc < width and frontier[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
        return clusters

    def _has_unexplored_area(self, cell):
        radius = max(1, int(self.frontier_radius / self.map_info.resolution))
        row, column = cell
        r0, r1 = max(0, row - radius), min(self.map_data.shape[0], row + radius + 1)
        c0, c1 = max(0, column - radius), min(self.map_data.shape[1], column + radius + 1)
        region = self.map_data[r0:r1, c0:c1]
        return region.size > 0 and float(np.count_nonzero(region == -1)) / region.size >= self.min_unexplored

    def _near_blocked(self, point):
        return any(math.dist(point, blocked) <= self.skip_radius for blocked in self.blocked_points)

    def _cluster_candidates(self, cluster, robot):
        """Return usable cells from a frontier cluster.

        A single connected frontier can surround most of the initially visible
        map.  Using only its centroid can therefore produce a goal almost at
        the robot's current pose.  Keep the individual frontier cells until
        after distance and blacklist checks so another part of that same
        cluster remains available after a goal is reached.
        """
        candidates = []
        for cell in cluster:
            point = self._cell_to_world(cell)
            if math.dist(point, robot) < self.min_goal_distance:
                continue
            if self._near_blocked(point):
                continue
            if self._has_unexplored_area(cell):
                candidates.append(point)
        return candidates

    def _try_next_goal(self):
        if self.goal_active or self.complete or self.map_data is None:
            return
        if not self.nav_client.server_is_ready():
            self._publish_state("waiting_for_nav2")
            return
        robot = self._robot_position()
        if robot is None:
            self._publish_state("waiting_for_tf")
            return
        clusters = self._find_frontier_clusters()
        candidates = []
        for cluster in clusters:
            candidates.extend(self._cluster_candidates(cluster, robot))
        if not candidates:
            # "No selectable goal" is not the same as "no frontier".  For
            # example, all currently visible frontier cells may be too close
            # to a recently visited/failed goal.  Reporting completion here
            # made the inspection begin immediately after its first short
            # navigation goal.
            if clusters:
                self.empty_map_updates = 0
                self.last_empty_map_version = -1
                self._publish_state("frontiers_temporarily_unreachable")
                self.get_logger().debug(
                    "Frontiers exist, but none is selectable yet; waiting for a map update."
                )
                return
            if self.map_version != self.last_empty_map_version:
                self.last_empty_map_version = self.map_version
                self.empty_map_updates += 1
            if self.empty_map_updates < self.empty_updates_required:
                self._publish_state("waiting_for_map_update")
                return
            self.complete = True
            self._publish_complete(True)
            self._publish_state("exploration_complete")
            self.get_logger().info("No viable frontiers remain; exploration is complete.")
            return
        self.empty_map_updates = 0
        self.last_empty_map_version = -1
        target = min(candidates, key=lambda point: math.dist(point, robot))
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = target[0]
        goal.pose.pose.position.y = target[1]
        goal.pose.pose.orientation.w = 1.0
        self.current_goal = target
        self.goal_active = True
        self._publish_complete(False)
        self._publish_state("exploring")
        self.get_logger().info(f"Sending frontier goal ({target[0]:.2f}, {target[1]:.2f})")
        self.nav_client.send_goal_async(goal).add_done_callback(self._goal_response)

    def _goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Could not send frontier goal: {exc}")
            self._finish_failed_goal()
            return
        if not handle.accepted:
            self.get_logger().warning("Frontier goal was rejected.")
            self._finish_failed_goal()
            return
        handle.get_result_async().add_done_callback(self._goal_result)

    def _goal_result(self, future):
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().warning(f"Frontier navigation failed: {exc}")
            self._finish_failed_goal()
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Reached frontier; refreshing the map.")
            # The occupancy grid can lag the robot by one or more update cycles.
            # Remember successful goals too, otherwise the same frontier may be
            # selected repeatedly while the map still contains its old cells.
            if self.current_goal is not None:
                self.blocked_points.append(self.current_goal)
            self.goal_active = False
            self.current_goal = None
        else:
            self.get_logger().warning(f"Frontier goal ended with status {status}; blacklisting it.")
            self._finish_failed_goal()

    def _finish_failed_goal(self):
        if self.current_goal is not None:
            self.blocked_points.append(self.current_goal)
        self.current_goal = None
        self.goal_active = False


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
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
