import argparse
import numpy as np
import os
import torch
import pdb
from tqdm import tqdm
import csv

from drone_pointpillars.utils import \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d, iou_bev
from drone_pointpillars.dataset import DroneDataset, get_dataloader
from drone_pointpillars.model import PointPillars 


def CsvWriteFramePredictions(csv_writer, frame_id, gt_boxes_3d, pred_boxes_3d, pred_labels, pred_scores):

    for gidx, gb in enumerate(gt_boxes_3d):
        # gb expected as [x,y,z,w,l,h,yaw] or bbox format you prefer
        csv_writer.writerow([frame_id, gidx, 'GT',
                            float(gb[0]), float(gb[1]), float(gb[2]),
                            float(gb[3]), float(gb[4]), float(gb[5]),
                            float(gb[6])])

    # write predicted rows with box_type = 'PRED'
    pred_num = pred_boxes_3d.shape[0] if pred_boxes_3d is not None else 0
    print(f"Amount of predictions in frame {frame_id}: {pred_num} ")
    for bidx in range(pred_num):
        bb = pred_boxes_3d[bidx]
        lbl = int(pred_labels[bidx]) if pred_labels.size else -1
        sc = float(pred_scores[bidx]) if pred_scores.size else 0.0
        csv_writer.writerow([frame_id, bidx, 'PRED',
                            float(bb[0]), float(bb[1]), float(bb[2]),
                            float(bb[3]), float(bb[4]), float(bb[5]),
                            float(bb[6]), lbl, sc])

def main(args):
    print("Loading dataset...")
    val_dataset = DroneDataset(data_root=args.data_root,
                        split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    
    # Setup the model and load checkpoint
    if args.model == 'simple':
        model = PointPillars(nclasses=args.nclasses).cuda()
    elif args.model == 'advanced':
        model = PointPillars(nclasses=args.nclasses, max_num_points=100, Backbone_layer_strides=[1,2,2],upsample_strides=[1,2,4]).cuda()
    
    print("Loading model...")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    # Choose the right key depending on how the checkpoint was saved
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
    model.eval()

    # Print model architecture
    print(model)

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    if args.csv:
        csv_path = os.path.join(saved_path, 'predictions.csv')
        csvfile = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_id','box_id','box_type','x','y','z','w','l','h','yaw','label','score'])

    """
    Steps in MAP evaluation:
    1. For each frame, get the predicted boxes and ground truth boxes. Then sort by confidence score
    2. Compute IoU between predicted boxes and ground truth boxes, and assign True Positive (TP) and False Positive (FP)
    3. Compute Precision-Recall curve and Average Precision (AP)
    """
    predictions = []
    gt_by_frame = {}

    ious = {
        'bbox_bev': [],
        'bbox_3d': []
    }

    IoU_thresholds = [0.5, 0.5]  # for bev and 3d

    '''
    # Box layout: [x, y, z, w, l, h, yaw]
    gt_box = torch.tensor([
        0.0,  # x
        0.0,  # y
        0.0,  # z
        2.0,  # w
        2.0,  # l
        2.0,  # h
        0.0   # yaw
    ])

    pred_box = torch.tensor([
        2.0 / 7.0,  # x shift ~0.2857
        0.0,        # y
        4.0 / 9.0,  # z shift ~0.4444
        2.0,        # w
        2.0,        # l
        2.0,        # h
        0.0         # yaw
    ])

    # This mimics the structure you use in your evaluation loop
    gts = [{"box": gt_box, "used_bev": False, "used_3d": False}]
    pred_box_3d_np = pred_box.numpy()  # your code expects numpy here
    
    gt_boxes_3d = torch.stack([g["box"] for g in gts]).float().cuda() # (num_gt, 7)
    gt_bev = gt_boxes_3d[:, [0, 1, 3, 4, 6]]                          # (num_gt, 5)
    
    pred_box_3d = torch.from_numpy(pred_box_3d_np).float().cuda()   # (7,)
    pred_box_3d_t = pred_box_3d.unsqueeze(0)                        # (1, 7)
    pred_bev = pred_box_3d_t[:, [0, 1, 3, 4, 6]]                    # (1, 5)

    # Calculate BEV IoU if we have gt and det boxes
    iou_bev_v = iou_bev(gt_bev, pred_bev) 
    iou3d_v = iou3d(gt_boxes_3d, pred_box_3d_t)
    print(f"BEV IoU: {iou_bev_v}, 3D IoU: {iou3d_v}")
    return
    ''' 
    
    # -----------------------------------------------------------
    # 1) For each frame, get the predicted and ground truth boxes 
    # -----------------------------------------------------------
    with torch.no_grad(): # no gradient computation, faster since we don't need it when evaluating
        print('Generating model predictions...')
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            # move the tensors to the cuda
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batch_results = model(batched_pts=batched_pts, mode='val')

            # Loop troguh batch results and save prediction(s) and ground truth(s)
            for j, result in enumerate(batch_results):
                frame_id = i * args.batch_size + j

                # ----------------- Predictions -----------------
                # Check if there where any predictions (dict) or not (empty lists)
                if isinstance(result, dict):
                    pred = result
                elif isinstance(result, (tuple, list)):
                    pred = {
                        'lidar_bboxes': result[0],
                        'labels': result[1],
                        'scores': result[2]
                    }
                    
                # ----------------- Save predictions for mAP evaluation -----------------
                for k in range(len(pred.get('scores', []))):
                    predictions.append({
                        "frame_id": frame_id,
                        "box": pred['lidar_bboxes'][k],  # (7,)
                        "score": pred['scores'][k].item()
                    })

                # ----------------- Save ground truth for mAP evaluation -----------------
                gt_boxes_3d = batched_gt_bboxes[j]  # torch tensor, shape (num_gt, 7) on CUDA
                gt_by_frame.setdefault(frame_id, [])
                for gt in gt_boxes_3d:         
                    gt_by_frame[frame_id].append({"box": gt, "used_bev": False, "used_3d": False})    

                # ------------ Save predictions and GT to CSV for local visualisation ------------ 
                if args.csv:
                    # convert torch tensor to numpy
                    pred_boxes_3d = np.asarray(pred.get('lidar_bboxes', []))  # (num_pred, 7)
                    pred_labels   = np.asarray(pred.get('labels', []))
                    pred_scores   = np.asarray(pred.get('scores', []))
                    # Ensure numpy arrays for preds
                    if torch.is_tensor(gt_boxes_3d):
                        gt_boxes_3d = gt_boxes_3d.cpu().numpy()
                    CsvWriteFramePredictions(csv_writer, frame_id, gt_boxes_3d, pred_boxes_3d, pred_labels, pred_scores)

    '''
    # 1) Keep only the highest-score prediction per frame
    best_by_frame = {}  # frame_id -> prediction dict
    for p in predictions:
        fid = p["frame_id"]
        if fid not in best_by_frame or p["score"] > best_by_frame[fid]["score"]:
            best_by_frame[fid] = p

    # 2) Replace predictions with the top-1 per frame
    predictions = list(best_by_frame.values())
    '''

    # Sort predictions by score descending
    predictions.sort(key=lambda p: -p["score"])

    # -----------------------------------------------------------------------------
    # 2) Compute IoU between predicted boxes and ground truth boxes and assign TP/FP
    # -----------------------------------------------------------------------------
    print("Calculating IoUs and TP/FP for each prediction...")

    # Precision-Recall init arrays
    N_pred = len(predictions)
    print(f"Total predictions: {N_pred}")
    TP_bev = np.zeros(N_pred, dtype=np.int32)
    FP_bev = np.zeros(N_pred, dtype=np.int32)
    TP_3d  = np.zeros(N_pred, dtype=np.int32)
    FP_3d  = np.zeros(N_pred, dtype=np.int32)

    #--- Errors ---
    x_errors = 0.0
    y_errors = 0.0
    z_errors = 0.0

    for i, pred in enumerate(tqdm(predictions)):
        frame_id = pred['frame_id']
        gts = gt_by_frame.get(frame_id, [])  # list of {"box": tensor(7,), "used": bool}
        pred_box_3d_np = pred['box']  # (7,)

        # If we have no GT boxes, all predictions are false positives
        if len(gts) == 0:
            FP_3d[i] = 1
            FP_bev[i] = 1
            continue

        # Calculate errors for matched predictions and ground truths
        x_errors += sum([abs(pred_box_3d_np[0] - g["box"][0].cpu().numpy()) for g in gts])
        y_errors += sum([abs(pred_box_3d_np[1] - g["box"][1].cpu().numpy()) for g in gts])
        z_errors += sum([abs(pred_box_3d_np[2] - g["box"][2].cpu().numpy()) for g in gts])

        # ----------------- Birds eye view IoU -----------------
        # Reduce dimensions of gt and pred [x, y, z, w, l, h, yaw]-> [x, y, w, l, yaw] for BEV IoU
        # (As we do not care about z and height for BEV IoU)
        gt_boxes_3d = torch.stack([g["box"] for g in gts]).float().cuda() # (num_gt, 7)
        gt_bev = gt_boxes_3d[:, [0, 1, 3, 4, 6]]                          # (num_gt, 5)
        
        pred_box_3d = torch.from_numpy(pred_box_3d_np).float().cuda()   # (7,)
        pred_box_3d_t = pred_box_3d.unsqueeze(0)                        # (1, 7)
        pred_bev = pred_box_3d_t[:, [0, 1, 3, 4, 6]]                    # (1, 5)

        # Calculate BEV IoU if we have gt and det boxes
        iou_bev_v = iou_bev(gt_bev, pred_bev)   # shape (num_gt, num_pred) or (num_pred, num_gt) depending on your order
        ious_flat = iou_bev_v.view(-1)     # (num_gt,) 1D vector with one IoU per GT box

        # Index of best GT for this prediction
        best_bev_idx = torch.argmax(ious_flat).item()     # integer
        best_bev_iou = ious_flat[best_bev_idx].item()     # Python float

        # Assign TP/FP based on IoU and whether GT box was already used
        if best_bev_iou >= IoU_thresholds[0] and not gts[best_bev_idx]["used_bev"]:
            TP_bev[i] = 1
            # Mark this GT box as used
            gts[best_bev_idx]["used_bev"] = True
        else:
            FP_bev[i] = 1

        # --------------------- 3D IoU ---------------------
        iou3d_v = iou3d(gt_boxes_3d, pred_box_3d_t) # (num_gt, 7), (1, 7) 
        ious_flat_3d = iou3d_v.view(-1)     # (num_gt,)
        ious['bbox_3d'].append(ious_flat_3d.cpu().numpy())

        # index of best GT for this prediction
        best_3d_idx = torch.argmax(ious_flat_3d).item()     # integer
        best_3d_iou = ious_flat_3d[best_3d_idx].item()      # Python float

        # Assign TP/FP based on IoU and whether GT box was already used
        if best_3d_iou >= IoU_thresholds[0] and not gts[best_3d_idx]["used_3d"]:
            TP_3d[i] = 1
            # Mark this GT box as used
            gts[best_3d_idx]["used_3d"] = True
        else:
            FP_3d[i] = 1
    
    # Average error print
    print(f"X error: {x_errors/N_pred}, Y error: {y_errors/N_pred}, Z error: {z_errors/N_pred}")

    # -------------------------------------------------------------
    # 3) Compute Precision-Recall curve and Average Precision (AP)
    # ------------------------------------------------------------- 
    print("Computing Precision-Recall curve and Average Precision (AP)...")
    # Total number of ground truth boxes
    N_gt = sum(len(v) for v in gt_by_frame.values())

    # -- BEV AP --
    len_TP = np.sum(TP_bev)
    len_FP = np.sum(FP_bev)
    print(f"Total GT boxes: {N_gt}, Total TP (BEV): {len_TP}, Total FP (BEV): {len_FP}")
    
    TP_cum = np.cumsum(TP_bev)
    FP_cum = np.cumsum(FP_bev)
    recall = TP_cum / N_gt
    precision = TP_cum / (TP_cum + FP_cum)
    # Ensure precision is non-increasing w.r.t recall (monotonic envelope)
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    AP_bev = np.trapz(precision, recall)
    print("BEV AP:", AP_bev)

    # -- 3D AP --
    len_TP = np.sum(TP_3d)
    len_FP = np.sum(FP_3d)
    print(f"Total GT boxes: {N_gt}, Total TP (3D): {len_TP}, Total FP (3D): {len_FP}")
    
    TP_cum = np.cumsum(TP_3d)
    FP_cum = np.cumsum(FP_3d)
    recall = TP_cum / N_gt
    precision = TP_cum / (TP_cum + FP_cum)
    # Ensure precision is non-increasing w.r.t recall (monotonic envelope)
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    AP_3d = np.trapz(precision, recall)
    print("3D AP:", AP_3d)

    if args.csv:
        csv_writer.writerow(['BEV_AP', AP_bev])
        #csv_writer.writerow(['3D_AP', AP_3d])
        csvfile.close()
        print(f"Saving predictions in CSV at path: {saved_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/ceph/project/P7_Gravel_Gun/Datasets/Drone_Data', 
                        help='your data root for drone dataset')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for drone dataset')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=1)
    parser.add_argument('--model', type=str, default='simple', help='model type')
    parser.add_argument('--csv', action='store_true', help='Whether to save the predictions in a csv file')

    args = parser.parse_args()

    main(args)
