"""Localization against a saved map: bridge + slam_toolbox localization + foxglove.

Usage: ros2 launch /slam/launch/slam_localization.launch.py map:=/slam/maps/<name>
(<name> = serialized posegraph basename, no extension)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map", description="posegraph basename, e.g. /slam/maps/lab"),
        ExecuteProcess(
            cmd=["python3", "/slam/bridge/zenoh_ros_bridge.py",
                 "--params", "/slam/bridge/bridge_params.yaml"],
            output="screen",
        ),
        Node(
            package="slam_toolbox",
            executable="localization_slam_toolbox_node",
            name="slam_toolbox",
            parameters=[
                "/slam/config/slam_localization.yaml",
                {"map_file_name": LaunchConfiguration("map")},
            ],
            output="screen",
        ),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"port": 8765, "address": "0.0.0.0"}],
            output="screen",
        ),
    ])
