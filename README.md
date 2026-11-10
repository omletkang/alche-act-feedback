# alche-act-feedback

**Alchemist Project (2024–2025)** · Soft Robotics and Bionics Lab (SRBL), Seoul National University

ROS1 package for collecting data from our **3-axis tactile force sensor** and a reference **load cell (RFT F/T sensor)**, and for calibrating the tactile sensor with a small **MLP neural network**.

The resulting sensor NN model converts the raw sensor signals into 3-axis force (Fx, Fy, Fz). We used it as the tactile input for an **ACT + tactile (3-axis force) fusion** imitation-learning model.

## Workflow

1. **Collect data**: record the raw 3-axis sensor signals together with the load cell's ground-truth force.
2. **Calibrate**: train an MLP (4 sensor inputs → 3-axis force) on the recorded CSV logs.
3. **Use the model**: run the trained model in ROS to publish calibrated force in real time.

## Contents

| Path | Description |
|---|---|
| `scripts/srbl/sensor_srbl.py` | Serial driver for the 3-axis finger sensor (ESP32, 2 sensors × 4 channels) |
| `scripts/sensor_node.py` | Publishes raw sensor data → `/sensor_data` |
| `scripts/sensor_force.py` | Runs the trained MLP and publishes calibrated force → `/sensor_force` |
| `scripts/sensor_2dplot*.py`, `sensor_3dplot*.py` | Real-time plots of the raw sensor data (`*_f`: calibrated force) |
| `scripts/gripper_node.py`, `gripper_feedback.py` | Dynamixel gripper control, including force feedback |
| `scripts/gello_node.py`, `src/record_gello.cpp`, `src/replay_gello.cpp` | GELLO teleoperation input, plus record/replay to CSV |
| `sensor_calibration/` | First MLP calibration (TensorFlow/Keras, `.h5`) |
| `sensor_cal_torch/`, `sensor_cal_torch3/`, `sensor_cal_torch_jh/` | PyTorch MLP calibration: `train.py`, data logs, trained models (`.pth`) and scalers (`.pkl`) |
| `launch/sensor_demo.launch` | Sensor node + 3D plot demo |

## Calibration data and model

- **CSV columns:** `RFT_1..RFT_6` (load cell Fx, Fy, Fz, Tx, Ty, Tz), then `data_1..data_4` (raw tactile sensor signals)
- **Model:** MLP `4 → 32 → 16 → 3` (ReLU), MSE loss, Adam optimizer, inputs normalized with `StandardScaler`

```bash
cd sensor_cal_torch3
python train.py   # set file_num inside; saves mlp_model_<n>.pth and scaler_<n>.pkl
```

## Run

```bash
# Ubuntu 20.04 + ROS Noetic, inside a catkin workspace
catkin_make && source devel/setup.bash
roslaunch alche_srbl sensor_demo.launch   # raw sensor + 3D plot
rosrun alche_srbl sensor_force.py         # calibrated force output
```
