# Beetle hardware bring-up

## Wiring used by the firmware

| Side | MDD20A direction pins | MDD20A PWM pins | Encoder used for odometry |
|---|---|---|---|
| Right | GPIO 5, GPIO 16 | GPIO 4, GPIO 15 | Rear-right: A GPIO 10, B GPIO 11 |
| Left | GPIO 7, GPIO 18 | GPIO 6, GPIO 17 | Front-left: A GPIO 8, B GPIO 9 |

The firmware drives the two motors on each side together. Encoder 5 V and GND must be common with ESP32 GND. Do **not** connect a 5 V encoder A/B signal directly to ESP32 GPIO; use a 3.3 V-compatible encoder output or a level shifter.

## Install and upload

Install `python3-serial` on the Pi and upload
`firmware/beetle_motor_controller/beetle_motor_controller.ino` with Arduino IDE or PlatformIO, selecting the correct ESP32-S3 board and USB port.

The Pi needs a stable device name for the ESP32. With both lidar and ESP32 connected, use `ls -l /dev/serial/by-id/` and create a udev rule or pass the correct ESP32 port through `serial_port:=...`. The supplied CP2102N `/dev/ttyUSB0` is the RPLIDAR, not assumed to be the ESP32.

## Start order

1. Start an RPLIDAR C1 ROS 2 driver configured to publish `/scan` and its static transform `base_link -> laser`.
2. Start a camera driver that publishes `/camera/image_raw`, `/camera/camera_info`, and `base_link -> camera_link`.
3. Build and launch the inspection stack:

```bash
cd /home/aditya/test/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch slam_based_nav beetle_inspection.launch.py serial_port:=/dev/ttyACM0
```

To enable damaged-product detection, provide a real-camera-trained YOLO model:

```bash
ros2 launch slam_based_nav beetle_inspection.launch.py \
  serial_port:=/dev/ttyACM0 \
  run_perception:=true \
  model_path:=/absolute/path/to/damaged_boxes_real.pt
```

Before Nav2, validate forward direction with the wheels raised:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05}, angular: {z: 0.0}}"
```

If a side drives backward or its encoder count decreases while moving forward, correct that side in firmware before autonomous motion. Tune `KP` and `KI`, then perform a one-wheel-revolution count test; correct `COUNTS_PER_WHEEL_REV` in both the firmware and bridge if it is not 5200.

The ESP32 stops both drivers after 300 ms without a valid command. It is a software fallback, not a substitute for a motor-power disconnect.
