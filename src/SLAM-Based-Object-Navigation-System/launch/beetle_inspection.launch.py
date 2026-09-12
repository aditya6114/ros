#!/usr/bin/env python3
"""Inspection stack for the physical Beetle base; start the RPLIDAR driver first."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory("slam_based_nav")
    slam_dir = get_package_share_directory("slam_toolbox")
    nav2_dir = get_package_share_directory("nav2_bringup")
    params = LaunchConfiguration("inspection_params")
    nav2_params = LaunchConfiguration("nav2_params")
    serial_port = LaunchConfiguration("serial_port")
    model_path = LaunchConfiguration("model_path")
    run_perception = LaunchConfiguration("run_perception")
    return LaunchDescription([
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("inspection_params", default_value=os.path.join(package_dir, "config", "inspection.yaml")),
        DeclareLaunchArgument("nav2_params", default_value=os.path.join(nav2_dir, "params", "nav2_params.yaml")),
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("run_perception", default_value="false"),
        Node(package="slam_based_nav", executable="beetle_serial_bridge.py", name="beetle_serial_bridge",
             parameters=[{"port": serial_port}], output="screen"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(slam_dir, "launch", "online_async_launch.py")),
                                 launch_arguments={"use_sim_time": "false"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav2_dir, "launch", "navigation_launch.py")),
                                 launch_arguments={"use_sim_time": "false", "params_file": nav2_params}.items()),
        Node(package="slam_based_nav", executable="object_mapper_lidar.py", name="object_mapper",
             parameters=[params, {"model_path": model_path}], output="screen",
             condition=IfCondition(run_perception)),
        Node(package="slam_based_nav", executable="object_reacher.py", name="go_to_object_node", parameters=[params], output="screen"),
        Node(package="slam_based_nav", executable="explorer2.py", name="frontier_explorer", parameters=[params], output="screen"),
    ])
