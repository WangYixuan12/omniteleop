# ROS2 Humble + slam_toolbox + zenoh(dexcomm) bridge for Vega-1 2D SLAM.
# Build (from repo root):
#   docker build -t yixuan-slam:humble SLAM/docker
# dexcomm is pinned to the robot-side version (bundles zenoh 1.8.0) so the
# wire format / zenoh protocol matches exactly. Do NOT add eclipse-zenoh.
FROM ros:humble-ros-base-jammy

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-slam-toolbox \
      ros-humble-nav2-map-server \
      ros-humble-foxglove-bridge \
      python3-pip

# numpy>=1.26 required by dexcomm; shadows Ubuntu's 1.21 (rclpy is agnostic).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install dexcomm==0.4.19 numpy==1.26.4

ENV ROBOT_NAME=dm/vg3cfe65689d-1
# Same absolute path as on the host; compose mounts the certs dir read-only.
ENV ZENOH_CONFIG=/home/dexmate/.dexmate/comm/zenoh/robopil/zenoh_peer_config.json5

WORKDIR /slam
