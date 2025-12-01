import argparse
import numpy as np
import os
from tqdm import tqdm
import sys
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from drone_pointpillars.utils import read_points, read_label, \
    write_pickle, get_points_num_in_bbox, write_points, points_in_bboxes_v2
    
def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)


def create_data_info_pkl(data_root, data_type, prefix, label=True, ids=None ,db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    # Okay so depending on whether it's train/val/test we read different files. They just have diffeent orders from 0-X. 

    split = 'training' if label else 'testing'

    drone_infos_dict = {}
    
    if db:
        # Create gt database for data augmentation
        drone_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)

        # We do not have a cam, so we use a fixed identity matrix for tr_velo_to_cam and r0_rect
        Tr_velo_to_cam_3x4 = np.array([
        [ 0., -1.,  0., 0.],
        [ 0.,  0., -1., 0.],
        [ 1.,  0.,  0., 0.],
        ], dtype=np.float32)

        tr_velo_to_cam = np.eye(4, dtype=np.float32)
        tr_velo_to_cam[:3, :] = Tr_velo_to_cam_3x4

        r0_rect = np.eye(4, dtype=np.float32)
        r0_rect[:3, :3] = np.eye(3, dtype=np.float32)  # 0_rect
  
    for id in tqdm(ids):
        cur_info_dict={}
        id_str = str(id).zfill(6)
        
        # Create directory to save points in gt bboxes for database augmentation
        lidar_path = os.path.join(data_root, split,'set3' ,'bin', f'{id_str}.bin')
        
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-4:])
        
        print(lidar_path)

        try:
            lidar_points = read_points(lidar_path)
        except:
            print(f'Error reading points from: {lidar_path}')
            break
    
        if label:
            label_path = os.path.join(data_root, split, 'set3', 'label', f'{id_str}.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=lidar_points,
                r0_rect=np.eye(4, dtype=np.float32), # Identity matrix for drone dataset (Used to be calib['R0_rect'])
                tr_velo_to_cam=np.eye(4, dtype=np.float32), # Identity matrix for drone dataset (Used to be calib['Tr_velo_to_cam'])
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

        drone_infos_dict[int(id)] = cur_info_dict

        if db:
            indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                points_in_bboxes_v2(
                    points=lidar_points,
                    r0_rect=r0_rect.astype(np.float32), 
                    tr_velo_to_cam=tr_velo_to_cam.astype(np.float32),
                    dimensions=annotation_dict['dimensions'].astype(np.float32),
                    location=annotation_dict['location'].astype(np.float32),
                    rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                    name=annotation_dict['name']    
                )
            for j in range(n_valid_bbox):
                db_points = lidar_points[indices[:, j]]
                db_points[:, :3] -= bboxes_lidar[j, :3]
                db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                write_points(db_points, db_points_saved_name)

                db_info={
                    'name': name[j],
                    'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                    'box3d_lidar': bboxes_lidar[j],
                    'difficulty': annotation_dict['difficulty'][j], 
                    'num_points_in_gt': len(db_points), 
                }
                
                if name[j] not in drone_dbinfos_train:
                    drone_dbinfos_train[name[j]] = [db_info]
                else:
                    drone_dbinfos_train[name[j]].append(db_info)
                    #print(db_info)
        
        id += 1

    if db:
        db_info_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(drone_dbinfos_train, db_info_path)
        print(f'Database info pkl file is saved to {db_info_path}')

    saved_path = os.path.join(data_root,split,'set3', f'{prefix}_infos_{data_type}.pkl')
    write_pickle(drone_infos_dict, saved_path)
    print(f'{data_type} data info pkl file is saved to {saved_path}')
    
    return drone_infos_dict

def main(args):
    data_root = args.data_root
    prefix = args.prefix

    # The dataset is split into train/val/test with a 70/15/15 ratio
    total = 56301 # Total number of samples in drone dataset
    all_ids = np.arange(total)

    # We shuffle the ids to create train/val/test splits
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    rng.shuffle(all_ids) # We shuffle everything.

    n_train = total * 70 // 100   # 70% Train
    n_val   = total * 15 // 100   # 15% Val
    n_test  = total - n_train - n_val  # remaining 15% Test

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    ## 1. train: create data infomation pkl file 
    ##           && create database(points in gt bbox) for data aumentation
    drone_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, ids=train_ids, db=True)

    ## 2. val: create data infomation pkl file 
    drone_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix, ids=val_ids, db=False)
    
    ## 3. trainval: create data infomation pkl file
    #drone_trainval_infos_dict = {**drone_train_infos_dict, **drone_val_infos_dict}
    #saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    #write_pickle(drone_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file
    drone_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, ids=test_ids, db=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/ceph/project/P7_Gravel_Gun/Datasets/Drone_Data', 
                        help='your data root for drone dataset')
    parser.add_argument('--prefix', default='drone', 
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()

    main(args)