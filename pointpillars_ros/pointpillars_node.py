import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, Quaternion
from std_msgs.msg import Header

import torch
import numpy as np
import math
from collections import OrderedDict

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from pointpillars_ros.drone_pointpillars.model import PointPillars
from pointpillars_ros.drone_pointpillars.utils import keep_bbox_from_lidar_range

class PointPillarsNode(Node):
    def __init__(self):
        super().__init__('pointpillars_node')

        # --- Parameters ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/livox/lidar') 
        self.declare_parameter('lidar_frame', 'livox_frame') 
        self.declare_parameter('world_frame', 'world') 
        self.declare_parameter('score_threshold', 0.3)
        self.declare_parameter('normalize_intensity', False)
        self.declare_parameter('model_type', 'large')
        
        # Correction for Training Height vs Real Height
        self.declare_parameter('ground_height_offset', 0.0) 
        
        # NEW: Manual Intensity Scaling (to match Sim "Red" color)
        self.declare_parameter('intensity_scale', 1.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.lidar_frame = self.get_parameter('lidar_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.score_thresh = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.normalize_intensity = self.get_parameter('normalize_intensity').get_parameter_value().bool_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        self.z_offset = self.get_parameter('ground_height_offset').get_parameter_value().double_value
        self.int_scale = self.get_parameter('intensity_scale').get_parameter_value().double_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Model Initialization ---
        if self.model_type == 'large':
            self.get_logger().info("Initializing LARGE (Advanced) PointPillars Model")
            self.model = PointPillars(nclasses=1, 
                                      max_num_points=100, 
                                      Backbone_layer_strides=[1,2,2], 
                                      upsample_strides=[1,2,4]).to(self.device)
        else:
            self.get_logger().info("Initializing SMALL (Simple) PointPillars Model")
            self.model = PointPillars(nclasses=1).to(self.device)
        
        if os.path.exists(model_path):
            self.load_pretrained_model(model_path)
        else:
            self.get_logger().error(f"Checkpoint not found at: {model_path}")

        self.model.eval()

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.listener_callback,
            10)
        
        self.publisher_ = self.create_publisher(MarkerArray, 'pointpillars_bbox', 10)
        
        #  RVIZ TO DEBUG ORIENTATION (It has corrected timestamps!)
        self.debug_pub_ = self.create_publisher(PointCloud2, '/debug/input_cloud', 10)
        
        self.get_logger().info(f"Node Init. Int Norm: {self.normalize_intensity}, Scale: {self.int_scale}, Z-Off: {self.z_offset}m")

    def load_pretrained_model(self, trained_path):
        ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        
        if any(k.startswith('module.') for k in state.keys()):
            state = OrderedDict((k.replace('module.', '', 1), v) for k, v in state.items())

        model_state = self.model.state_dict()
        filtered_state = OrderedDict()
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
        
        self.model.load_state_dict(filtered_state, strict=False)
        self.get_logger().info("Checkpoint loaded successfully.")

    def get_transform_matrix(self, target_frame, source_frame):
        try:
            # FIX: Use Time() (zero) to get the LATEST available transform.
            # asking for specific 'now' often causes ExtrapolationExceptions due to latency.
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().warn(f"Waiting for TF {target_frame} -> {source_frame}...", throttle_duration_sec=2.0)
                return None, None, None

            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            
            trans = np.array([t.transform.translation.x, 
                              t.transform.translation.y, 
                              t.transform.translation.z])

            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            
            x, y, z, w = q
            R = np.array([
                [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
                [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
                [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
            ])
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = trans
            
            return T, R, trans

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF Error: {e}")
            return None, None, None

    def get_yaw_from_matrix(self, R):
        return np.arctan2(R[1, 0], R[0, 0])

    def listener_callback(self, msg):
        # --- TIME SYNC FIX ---
        # Overwrite timestamp to NOW so it matches the TF system time
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()

        # 1. Get Lidar -> World Transform (Latest available)
        _, R_lidar_to_world, T_lidar_to_world = self.get_transform_matrix(
            self.world_frame, 
            self.lidar_frame)
        
        if R_lidar_to_world is None:
            return

        # 2. Extract Drone Heading (Yaw)
        yaw_global = self.get_yaw_from_matrix(R_lidar_to_world)
        
        # 3. Create "Inverse Yaw" Matrix (Stabilizer)
        c, s = np.cos(-yaw_global), np.sin(-yaw_global)
        R_world_to_stabilized = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # 4. Compute Final Transform: Lidar -> Stabilized
        R_lidar_to_stabilized = np.dot(R_world_to_stabilized, R_lidar_to_world)

        # 5. Read Points
        points_list = []
        try:
            for point in point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
        except Exception:
             return

        if not points_list:
            return

        intensities = []
        try:
            for p in point_cloud2.read_points(msg, field_names=['intensity'], skip_nans=True):
                val = p[0]
                # Apply Scaling first
                val = val * self.int_scale
                
                if self.normalize_intensity:
                    val = val / 255.0
                intensities.append(val)
        except Exception:
            intensities = [0.0] * len(points_list)

        pc_lidar = np.array(points_list, dtype=np.float32) 
        
        # 6. Apply Rotation (Lidar -> Stabilized)
        pc_aligned = np.dot(R_lidar_to_stabilized, pc_lidar.T).T
        
        # 7. Apply Z-Correction
        pc_aligned[:, 2] += self.z_offset
        
        # Stack intensity
        pc_input = np.hstack((pc_aligned, np.array(intensities).reshape(-1, 1)))

        # --- DEBUG VISUALIZATION ---
        if self.debug_pub_.get_subscription_count() > 0:
            header = Header()
            header.frame_id = self.world_frame 
            header.stamp = msg.header.stamp
            
            # Rotate back to World
            R_stabilized_to_world = R_world_to_stabilized.T 
            
            # Remove Z-Correction for visualization
            debug_pts_input_frame = pc_aligned.copy()
            debug_pts_input_frame[:, 2] -= self.z_offset
            
            # Rotate and Translate to World
            debug_pts_world = np.dot(R_stabilized_to_world, debug_pts_input_frame.T).T
            debug_pts_world += T_lidar_to_world
            
            debug_data = np.hstack((debug_pts_world, np.array(intensities).reshape(-1, 1)))
            
            debug_msg = point_cloud2.create_cloud(header, [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ], debug_data)
            self.debug_pub_.publish(debug_msg)
        # ---------------------------

        # Filter
        pcd_limit_range = np.array([0, -40, -40, 70, 40, 40], dtype=np.float32)
        mask = (pc_input[:, 0] > pcd_limit_range[0]) & (pc_input[:, 0] < pcd_limit_range[3]) & \
               (pc_input[:, 1] > pcd_limit_range[1]) & (pc_input[:, 1] < pcd_limit_range[4]) & \
               (pc_input[:, 2] > pcd_limit_range[2]) & (pc_input[:, 2] < pcd_limit_range[5])
        pc_input = pc_input[mask]

        if pc_input.shape[0] == 0:
            return

        # Inference
        pc_tensor = torch.from_numpy(pc_input.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            result = self.model(batched_pts=[pc_tensor], mode='test')

        if isinstance(result, list):
            result = result[0]
        if isinstance(result, tuple):
            return 

        lidar_bboxes = result['lidar_bboxes'] 
        scores = result['scores']

        # Publish Markers
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # R_z(yaw_global)
        c_back, s_back = np.cos(yaw_global), np.sin(yaw_global)
        R_stabilized_to_world = np.array([
            [c_back, -s_back, 0],
            [s_back,  c_back, 0],
            [0,       0,      1]
        ])

        for i in range(len(scores)):
            if scores[i] < self.score_thresh:
                continue

            bbox = lidar_bboxes[i]
            
            # 1. Get box center in Stabilized Frame
            pos_stabilized = np.array([bbox[0], bbox[1], bbox[2]])
            
            # 2. Remove Z-Correction
            pos_stabilized[2] -= self.z_offset
            
            # 3. Rotate to World
            pos_world_rel = np.dot(R_stabilized_to_world, pos_stabilized)
            
            # 4. Add Drone Position
            pos_world = pos_world_rel + T_lidar_to_world

            # Orientation
            yaw_bbox = bbox[6]
            final_yaw = yaw_bbox + yaw_global
            q_final = self.yaw_to_quaternion(final_yaw)

            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = msg.header.stamp
            marker.ns = "pointpillars"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(pos_world[0])
            marker.pose.position.y = float(pos_world[1])
            marker.pose.position.z = float(pos_world[2])
            marker.pose.orientation = q_final

            marker.scale.x = float(bbox[3])
            marker.scale.y = float(bbox[4])
            marker.scale.z = float(bbox[5])

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5 

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 50000000 

            marker_array.markers.append(marker)

        self.publisher_.publish(marker_array)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = PointPillarsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()