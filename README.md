# Drone Moving Platform Takeoff Setup

## Overview
This package enables precise takeoff from moving platforms using:
- OAK-D Pro (downward-facing) for AprilTag detection and VIO
- Jetson Nano (Ubuntu 18.04) for processing
- CUAV V5+ flight controller
- Benewake TFmini for altitude

## Hardware Setup
1. Mount OAK-D Pro downward-facing on drone
2. Connect OAK-D Pro to Jetson Nano via USB3
3. Connect TFmini to CUAV V5+ via UART:
   - TFmini TX → CUAV V5+ RX
   - TFmini RX → CUAV V5+ TX
   - TFmini GND → CUAV V5+ GND
   - TFmini VCC → CUAV V5+ 5V
4. Mount AprilTag (tag36h11, 10cm) on platform

## Software Installation
1. Run installation script:
   ```bash
   chmod +x scripts/install_dependencies.sh
   ./scripts/install_dependencies.sh

## Launch
roslaunch drone.launch
