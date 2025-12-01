import torch 
import numpy as np
from drone_pointpillars.model import PointPillars
from drone_pointpillars.utils import keep_bbox_from_lidar_range

def point_range_filter(pts, point_range=[0, -40, -40, 70, 40, 40]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def load_model_checkpoint(trained_path,model):
    ckpt = torch.load(trained_path)

    # Load weights
    if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state = ckpt['model']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            else:
                # assume whole dict is a state_dict
                state = ckpt
    else:
        # old-style: ckpt *is* the state_dict
        state = ckpt

    model.load_state_dict(state)

def main():
    point_cloud = {}  # Replace with actual point cloud data loading

    # Parameters
    pcd_limit_range = np.array([0, -40, -40, 70, 40, 40], dtype=np.float32)

    # Init model (Choose small or large model)
    if True:
        # Ssmall model 
        trained_path = "/home/antonio/workspaces/pointpillars_UAV_ros2_ws/models_and_datasets/APointPillars_Results_10/checkpoints/epoch_10.pth" 
        model = PointPillars(nclasses=1).cuda()
    else:
        # Large model
        trained_path = "/home/antonio/workspaces/pointpillars_UAV_ros2_ws/models_and_datasets/SPointPillars_Results_15/checkpoints/epoch_15.pth"
        model = PointPillars(nclasses=1, max_num_points=100, Backbone_layer_strides=[1,2,2],upsample_strides=[1,2,4]).cuda()
        
    load_model_checkpoint(trained_path, model)
    model.eval()

    with torch.no_grad():
        while True:
            # Remove points from point cloud which is out of range 
            pc = point_range_filter(pc, pcd_limit_range)

            # Run the model
            result = model(batched_pts=pc, mode='test')

            # Remove predictions which is out of range
            result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

            # Split up results
            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']

main()