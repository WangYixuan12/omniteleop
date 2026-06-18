Sensors
IMU

Two embedded IMUs within two 3D Lidars.

An IMU within the head camera.

Lidar

Two 3D Lidars from RoboSense on both the front and the back of the chassis 


Ultrasonic sensor(Vega-1 Only)
Demonstrates how to retrieve distance measurements from the robot's ultrasonic sensor.

```bash
from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config

configs = get_robot_config()
configs.sensors.ultrasonic.enable = True
robot = Robot(configs=configs)

distance = robot.sensors.ultrasonic.get_obs()
print(f"Ultrasonic sensor distance (m): {distance}")

robot.shutdown()
```

 Lidar data
Demonstrates how to get the Lidar scan data from the robot

```bash
# launch dexsensor on Jetson
dexsensor launch --sensor lidar

Copy
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

configs = get_robot_config()
configs.sensors.lidar.enable = True
robot = Robot(configs=configs)

scan_data = robot.sensors.lidar.get_obs()

ranges = scan_data["ranges"]
angles = scan_data["angles"]
qualities = scan_data.get("qualities")

print(f"Ranges: {ranges}")
print(f"Angles: {angles}")
print(f"Qualities: {qualities}")

robot.shutdown()
```

Sensor Library
Our robotic platform features multiple sensors including cameras, LiDAR, and other perception devices. Some of these sensors are directly connected to the Jetson hardware, while others may be connected through different interfaces. While users have the flexibility to write custom code for direct sensor access, we provide a comprehensive sensor library that streamlines this process.

The dexsensor library handles the low-level sensor communication and offers network streaming capabilities, allowing you to access sensor data easily.

dexsensor can be downloaded from https://software.dexmate.ai/packages/dexsensor

The above link provides the library packed into a debian that is optimized for Jetson Thor or other arm or amd machines. One must be careful in downloading the right version of the library before installing.

The following commands need to be run after downloading the appropriate debian to install the library.


Copy
sudo dpkg -i <dexsensor_xxx>.deb
WARNING

If you have a pre-existing version of python based dexsensor and want to upgrade to our latest library, it is advisable to first cleanly uninstall dexsensor<=0.3.1 and then install dexsensor>=0.6.9 .

NOTICE

We pre-installed the ZED SDK on all the Jetsons we ship, located at /usr/local/zed. The Python API for the ZED SDK is also installed in the base Conda environment. If you do not find PyZED installed or need to install it in your own Python environment, you can go to /usr/local/zed and run python get_python_api.py to install PyZED.


If you encounter a "ZED SDK not found" error when launching dexsensor, but can successfully import pyzed in a Python terminal, this indicates a library compatibility issue. Fix it by updating the C++ standard library: conda update -c conda-forge libstdcxx-ng -y .

Quick Start Guide for dexsensor
After installing dexsensor, you can launch the sensor suite on your Jetson device using the command-line interface.

👓 Available Sensors
You can specify any combination of the following sensor names:

head_camera

base_left_camera

base_right_camera

base_front_camera

base_back_camera

lidar_3d_front/lidar_2d_front

lidar_3d_back

If you set --sensor all , it will launch all the sensors mentioned above. If you set --sensor base_camera, it will launch all four cameras on the robot chassis.

Note: Vega-1 comes with 2D lidar (on Jetson Orin) and Vega-1-Pro comes with 3D lidar (on Jetson Thor). The latest dexsensor>=0.6.9 can support both.

Important Note:

The sensor suite is available entirely on Jetson Orin on Vega-1, while it is divided amongst two edge computers on Vega-1-Pro. The distribution is as follows:

Jetson Thor
Jetson Nano
Base USB cameras

Head camera

Lidars

Wrist camera

🧺 Basic Launch
To start specific sensors, use the launch command with the --sensor parameter:


Copy
# Launch the sensor suite with selected sensors on respective Jetson
dexsensor launch --sensor <sensor_names>

# Example: Launch head camera (Jetson Nano on Vega-1-Pro and Jetson Orin on Vega-1)
dexsensor launch --sensor head_camera --sensor base_front_camera

# Example: Launch front lidar (Jetson Thor on Vega-1-Pro)
dexsensor launch --sensor lidar_front

# Generate the default configuration file
# Saves to ~/.dexmate/sensors/default_config.yaml by default
dexsensor gen-cfg

# Launch with your custom configuration
dexsensor launch --config ~/.dexmate/sensors/default_config.yaml