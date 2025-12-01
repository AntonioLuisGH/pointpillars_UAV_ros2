# Counter_UAV_System
Counter_UAV_System is a multi-sensor, close-range UAV interception platform that integrates acoustic detection, LiDAR-based localization, and predictive targeting to neutralize drones using a pneumatic gun.
The system uses a tetrahedral 4-microphone array for acoustic UAV detection and classification via Time Difference of Arrival (TDOA) and machine learning. A 3D conical LiDAR is used for precise localization and classification of UAVs using the PointPillars architecture. A predictive model estimates UAV velocity and future position to enable accurate aiming and firing of the pneumatic gun when the UAV is within range.

## 🛡️ Overview
**Counter_UAV_System** is a close-range UAV interception platform that integrates acoustic and LiDAR-based sensing with predictive targeting to neutralize drones using a pneumatic gun.

## 🚀 Key Features
- **Acoustic Detection**: Tetrahedral 4-microphone array for UAV detection using Time Difference of Arrival (TDOA) and acoustic classification.
- **LiDAR Localization**: 3D conical LiDAR with PointPillars for UAV classification and precise localization.
- **Predictive Targeting**: Motion model estimates UAV velocity and future position to aim and fire the pneumatic gun accurately.
- **Interception Logic**: Fires only when UAV is within range and predicted to intersect the gun’s trajectory.


## 🛠️ Installation
```bash
git clone https://github.com/your-username/Counter_UAV_System.git
cd Counter_UAV_System
# Add setup instructions here
