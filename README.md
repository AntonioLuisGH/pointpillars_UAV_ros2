# pointpillars_ros

This ROS2 package provides a real-time implementation of PointPillars for 3D object detection, specifically tuned for detecting small UAVs (Drones) using data from a conical LiDAR (e.g., Livox Avia).

This software was developed as part of a **Counter-UAV System project at Aalborg University**.

## 📋 Project Context

This package is a core component of a larger UAV tracking and neutralization system. The full architecture operates as follows:

1.  **Detection:** An Acoustic Sensor Array estimates the Direction of Arrival (DoA) of an incoming drone.
2.  **Tracking:** A Pan-Tilt Unit (PTU) uses the DoA to physically point a narrow-FOV conical LiDAR towards the target.
3.  **Classification (This Package):** The LiDAR captures point clouds, and this PointPillars node detects and estimates the 3D bounding box of the drone.
4.  **Fusion:** A Particle Filter fuses acoustic and spatial data to maintain a lock.
5.  **Neutralization:** A pneumatic gravel cannon is targeted based on the estimated position.

## ✨ Key Features

*   **Real-time Inference:** Runs PointPillars efficiently on CUDA-enabled GPUs.
*   **Tilt Stabilization ("Virtual Stabilized Frame"):**
    *   Since the LiDAR is often pitched up (30°-60°) to track airborne targets, standard models often fail.
    *   This node mathematically rotates the point cloud to be "flat" (gravity-aligned) before inference, ensuring standard PointPillars models work correctly without needing to be retrained for specific tilt angles.
*   **ROS2 Integration:** Fully integrated with the ROS2 ecosystem (`rclpy`, `sensor_msgs`, `visualization_msgs`, `tf2`).
*   **Custom Visualization:** Publishes detection results as `visualization_msgs/MarkerArray` (Yellow 3D Cubes) for easy verification in Rviz2.

## 🛠️ Dependencies

### System Requirements
*   **ROS2** (Humble or newer recommended)
*   **CUDA Toolkit** (Required for compiling PointPillars extensions)
*   **Python 3.10+**

### Python Libraries
*   `torch` (PyTorch with CUDA support)
*   `numpy`
*   `numba`
*   `open3d` (optional, for visualization scripts)

## 📥 Installation

1.  **Clone the repository** into your ROS2 workspace `src` folder:
    ```bash
    cd ~/your_ws/src
    git clone <repository_url> pointpillars_ros
    ```

2.  **Install ROS Dependencies:**
    ```bash
    sudo apt install ros-humble-vision-msgs ros-humble-tf2-sensor-msgs
    ```

3.  **Build the Package:**
    > **Note:** This package builds C++/CUDA extensions during installation. It may take a few minutes.
    
    ```bash
    cd ~/your_ws
    colcon build --packages-select pointpillars_ros --event-handlers console_direct+
    ```

4.  **Source the workspace:**
    ```bash
    source install/setup.bash
    ```

## 🚀 Usage

Run the detection node using `ros2 run`. You must provide the absolute path to your trained model checkpoint.

```bash
ros2 run pointpillars_ros detect --ros-args \
  -p model_path:=/path/to/your/checkpoint.pth \
  -p input_topic:=/lidar/avia \
  -p lidar_frame:=lidar_link \
  -p world_frame:=drone_world \
  -p model_type:=small \
  -p normalize_intensity:=False
```

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `string` | **Required** | Absolute path to the `.pth` checkpoint file. |
| `model_type` | `string` | `small` | Architecture type. Options: `'small'` (standard) or `'large'` (advanced backbone). |
| `input_topic` | `string` | `/lidar/avia` | The ROS2 topic for input `PointCloud2` messages. |
| `lidar_frame` | `string` | `lidar_link` | The TF frame attached to the physical sensor. |
| `world_frame` | `string` | `drone_world` | A gravity-aligned fixed frame (Z-up). Used for stabilization. |
| `score_threshold` | `float` | `0.3` | Confidence threshold for publishing detections. |
| `normalize_intensity`| `bool` | `False` | Set `True` if model expects 0.0-1.0 intensity. Set `False` if trained on raw 0-255. |

### Subscribed Topics

*   **Input PointCloud:** `sensor_msgs/PointCloud2`
    *   Defined by the `input_topic` parameter.
*   **TF:** `/tf` and `/tf_static`
    *   Required for the stabilization logic.

### Published Topics

*   **Detections:** `/pointpillars_bbox` (`visualization_msgs/MarkerArray`)
    *   Visualizes detected drones as semi-transparent yellow cubes.
*   **Debug Cloud:** `/debug/input_cloud` (`sensor_msgs/PointCloud2`)
    *   Shows the exact point cloud seen by the neural network (stabilized and aligned). Useful for debugging TF/orientation issues.

## 🔗 Related Repositories

The dataset generation and training pipeline for this project utilizes the following repositories:

*   **Simulation Environment:** `drone_detector_sim` - ROS package for generating synthetic training data.
*   **Data Conversion:** `bag_to_kitti_converter` - Tools to convert ROS2 bags into the KITTI format required for training.
*   **Model Training:** `Counter_UAV_System` (PointPillars Adaption) - The training code used to generate the `.pth` checkpoints.
