import os
import torch
import numpy as np
from drone_pointpillars.model import PointPillars
from drone_pointpillars.utils import keep_bbox_from_lidar_range
from drone_pointpillars.utils import drone_io

def point_range_filter(pts, point_range=[0, -40, -40, 70, 40, 40]):
    '''
    pts: numpy array of points
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

def load_model_checkpoint(trained_path, model):
    if not os.path.exists(trained_path):
        print(f"Error: Checkpoint not found at {trained_path}")
        return

    ckpt = torch.load(trained_path, map_location='cpu')

    # Load weights
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

    # Load state dict
    model.load_state_dict(state, strict=False)
    print("Checkpoint loaded successfully.")

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the uploaded demo point cloud file
    pcd_path = os.path.join(script_dir, 'drone_pointpillars', 'dataset', 'demo_data', 'test', '000002.bin')
    
    # Path to the uploaded model checkpoint
    trained_path = os.path.join(script_dir, 'pretrained', 'epoch_160.pth')

    if not os.path.exists(pcd_path):
        print(f"File not found: {pcd_path}")
        return

    # Load the point cloud using the utility function
    pc = drone_io.read_points(pcd_path)
    print(f"Loaded point cloud with {pc.shape[0]} points.")

    # Parameters
    pcd_limit_range = np.array([0, -40, -40, 70, 40, 40], dtype=np.float32)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CHANGED: nclasses=3 to match the checkpoint weights (18 channels)
    model = PointPillars(nclasses=3).to(device)
    
    load_model_checkpoint(trained_path, model)
    model.eval()

    with torch.no_grad():
        # Remove points from point cloud which is out of range 
        pc = point_range_filter(pc, pcd_limit_range)

        # Convert to tensor and move to device
        pc_tensor = torch.from_numpy(pc).to(device)

        # Run the model
        # The model expects a list of tensors for the batch
        result = model(batched_pts=[pc_tensor], mode='test')

        # result is likely a list (batch size 1), get the first item
        if isinstance(result, list):
            result = result[0]

        # Remove predictions which are out of range
        result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

        # Split up results
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        
        print("\n--- Inference Results ---")
        print(f"Detected {len(labels)} objects.")
        if len(labels) > 0:
            print("Scores:", scores.cpu().numpy())
            print("Labels:", labels.cpu().numpy())
            print("BBoxes (x, y, z, w, l, h, rot):\n", lidar_bboxes.cpu().numpy())
        else:
            print("No objects detected in this frame.")

if __name__ == '__main__':
    main()