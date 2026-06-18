"""Online mapping: zenoh bridge + slam_toolbox (async) + foxglove ws on :8765."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        ExecuteProcess(
            cmd=["python3", "/slam/bridge/zenoh_ros_bridge.py",
                 "--params", "/slam/bridge/bridge_params.yaml"],
            output="screen",
        ),
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            parameters=["/slam/config/slam_mapping.yaml"],
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
