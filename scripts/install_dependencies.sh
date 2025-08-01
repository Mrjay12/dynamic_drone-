#!/bin/bash

# Install ROS Melodic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-melodic-ros-base ros-melodic-mavros ros-melodic-apriltag-ros python3-venv python3-pip

# Install DepthAI
python3 -m venv depthai-env
source depthai-env/bin/activate
pip install depthai==2.19.0

# Install depthai-ros
cd ~/catkin_ws/src
git clone -b melodic https://github.com/luxonis/depthai-ros.git
cd ../
catkin_make

# Create ROS package
cd ~/catkin_ws/src
catkin_create_pkg unified_drone_system rospy geometry_msgs mavros_msgs apriltag_ros tf nav_msgs std_msgs
cd ../
catkin_make

# Make node executable
chmod +x ~/catkin_ws/src/unified_drone_system/scripts/unified_flight_controller.py

echo "Installation complete! Reboot your system."
