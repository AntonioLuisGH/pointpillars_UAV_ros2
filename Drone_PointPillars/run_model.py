import os
import torch
import numpy as np
from collections import OrderedDict
from drone_pointpillars.model import PointPillars
from drone_pointpillars.utils import keep_bbox_from_lidar_range
from drone_pointpillars.utils import drone_io

def point_range_filter(pts, point_range=[0, -40, -40, 70, 40, 40]):
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def load_pretrained_model(model, trained_path):
    if not os.path.exists(trained_path):
        print(f"Error: Checkpoint not found at {trained_path}")
        return

    # Load with weights_only=False to support older checkpoints
    ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
        
    if any(k.startswith('module.') for k in state.keys()):
        state = OrderedDict((k.replace('module.', '', 1), v) for k, v in state.items())

    model_state = model.state_dict()
    filtered_state = OrderedDict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            if k in model_state:
                print(f'[Warning] Skipping {k}: Checkpoint {tuple(v.shape)} vs Model {tuple(model_state[k].shape)}')

    model.load_state_dict(filtered_state, strict=False)
    print("Checkpoint loaded successfully (partial load).")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    pcd_path = os.path.join(script_dir, 'drone_pointpillars', 'dataset', 'demo_data', 'test', '000002.bin')
    trained_path = "/home/antonio/workspaces/pointpillars_UAV_ros2_ws/models_and_datasets/APointPillars_Results_10/checkpoints/epoch_10.pth"

    if not os.path.exists(pcd_path):
        print(f"File not found: {pcd_path}")
        return

    pc = drone_io.read_points(pcd_path)
    print(f"Loaded point cloud with {pc.shape[0]} points.")

    pcd_limit_range = np.array([0, -40, -40, 70, 40, 40], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PointPillars(nclasses=1).to(device)
    load_pretrained_model(model, trained_path)
    model.eval()

    with torch.no_grad():
        pc = point_range_filter(pc, pcd_limit_range)
        pc_tensor = torch.from_numpy(pc).to(device)

        result = model(batched_pts=[pc_tensor], mode='test')

        if isinstance(result, list):
            result = result[0]

        result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        
        print("\n--- Inference Results ---")
        print(f"Detected {len(labels)} objects.")
        if len(labels) > 0:
            # FIX: Removed .cpu().numpy() because these are already numpy arrays
            print("Scores:", scores)
            print("Labels:", labels)
            print("BBoxes:\n", lidar_bboxes)
        else:
            print("No objects detected.")

if __name__ == '__main__':
    main()