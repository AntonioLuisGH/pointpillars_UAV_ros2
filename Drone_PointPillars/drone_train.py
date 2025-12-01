import argparse
import os
import torch
from tqdm import tqdm # Progress bar
import pdb
import time

from drone_pointpillars.utils import setup_seed
from drone_pointpillars.dataset import drone_set, get_dataloader 
from drone_pointpillars.model import PointPillars
from drone_pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter

# To use multipule GPUS
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

def load_pretrained_model(pointpillars, is_main, ckpt):    

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
        
    # strip "module." if checkpoint came from DDP. DDP addes module to the name of each parameter.
    if any(k.startswith('module.') for k in state.keys()):
        state = OrderedDict((k.replace('module.', '', 1), v) for k, v in state.items())

    # Keep only parameters that match in name and size
    model_state = pointpillars.state_dict()
    filtered_state = OrderedDict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            # Logs both wrong-shape and completely unknown keys
            if is_main:
                print(f'[skip] {k}: ckpt {tuple(v.shape)} vs model {tuple(model_state.get(k, torch.empty(0)).shape)}')

    msg = pointpillars.load_state_dict(filtered_state, strict=False)  # (Strictload) False: gives warning if keys do not match. True: Gives error and stops.
    if is_main:
        print(f'[#Successfully loaded pretrained model#]',flush=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def main_workers(rank, world_size, args):
    training_start_time = time.time()

    # --------------- DDP setup ---------------
    torch.cuda.empty_cache() # free cached memory (I had an issue where it ran out of memory)
    setup(rank,world_size) # This setup is to use multipule GPUS.
    is_main = (rank == 0)  # Check if this is the main process / GPU nr 0
    if is_main:
        print(f"Starting training... GPUs:{world_size}, Batch size per GPU: {args.batch_size}, Max epochs: {args.max_epoch}, Checkpoint freq (epochs): {args.ckpt_freq_epoch}",flush=True)

    # --------------- Dataset and Dataloader ---------------
    train_dataset = drone_set.DroneDataset(data_root=args.data_root,
                          split='train')
    val_dataset = drone_set.DroneDataset(data_root=args.data_root,
                        split='val')

    # We have two datasets, one for training and one for validation.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = get_dataloader(dataset=train_dataset,    
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,            # sampler handles it
                                  sampler=train_sampler)

    val_dataloader = get_dataloader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,
                                sampler=val_sampler)
    
    # --- Initialize the model weights ---
    setup_seed() # First all are randomly initialized

    if args.model == 'simple':
        # Pointpillars with only the head changed (Amount of clases)
        pointpillars = PointPillars(nclasses=args.nclasses).to(rank)
    elif args.model == 'advanced':
        # PointPillars but with more points per pillar, and a backbone with finer spatial resolution. 
        pointpillars = PointPillars(nclasses=args.nclasses, max_num_points=100, Backbone_layer_strides=[1,2,2],upsample_strides=[1,2,4]).to(rank)
    
    # Then we can try an load the pretrained model (We load only wheights that match)
    ckpt = torch.load(args.pretrained, map_location='cpu')
    if is_main:
        print(f'[Loading pretrained model from {args.pretrained}...]', flush=True)
    if args.pretrained != 'None':
        load_pretrained_model(pointpillars, is_main, ckpt)

    loss_func = Loss()
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr

    # Freeze model wheights, and unfreeze detection head.
    if args.freeze:
        print("Freezing entire model", flush=True)
        for param in pointpillars.parameters():
            param.requires_grad = False

        for param in pointpillars.head.parameters():
            param.requires_grad = True
            print("Unfreezing parameter:", param.shape, flush=True)

        optimizer = torch.optim.AdamW([p for p in pointpillars.parameters() if p.requires_grad], lr=init_lr, betas=(0.95, 0.99), weight_decay=0.01)
    else:
        # Optmizer if the whole model is being trained
        optimizer = torch.optim.AdamW(pointpillars.parameters(), lr=init_lr, betas=(0.95, 0.99), weight_decay=0.01) # Updated to fit new DDP model

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    
    if args.pretrained != 'None':
        try:
            # Load optimizer state
            optimizer.load_state_dict(ckpt["optimizer"])
            # Load scheduler state
            scheduler.load_state_dict(ckpt["scheduler"])
            # Load epoch counter
            start_epoch = ckpt["epoch"] + 1

        except Exception as e:
            start_epoch = 0
            if is_main:
                print(f"[Warning] Could not load optimizer and scheduler state: {e}", flush=True)   
    else:
        start_epoch = 0

    # We wrap the model in DDP, to use multipule GPUs 
    pointpillars = torch.nn.parallel.DistributedDataParallel(pointpillars, device_ids=[rank], output_device=rank, find_unused_parameters=False)
       
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(start_epoch, args.max_epoch):
        start_time = time.time()
        train_sampler.set_epoch(epoch) # ADDED, allows shuffling with DDP
        train_step, val_step = 0, 0

        print("GPU ", rank, " starting epoch ", epoch+1, "/", args.max_epoch, flush=True)

        for i, data_dict in enumerate(tqdm(train_dataloader,disable=not is_main)): # Only show progress bar on main process
            
            # (Moves every tensor in the batch from CPU → GPU memory.)
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = item.to(rank, non_blocking=True)   # Changed to non_blocking=True so that multipule GPUs can work faster

            # ---- Standard step: clear old grads, run the model, then flatten/permute the BEV feature maps so predictions align with anchor targets. ----
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes, 
                             batched_gt_labels=batched_labels)
            
            # Flatten the entire batch into one big tensor for loss computation
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            # ----- We select only positive anchors (matched to a GT box) for the regression and direction losses. -----
            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1) 
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]

            # --- Implements the stable loss on angles via sin(a-b) identity to avoid wrap-around at π/−π. --- 
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            # ----- Keep anchors with positive label weight for classification; convert ignored anchors to the “background” bin for loss implementation.
            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            # --- Compute losses  ---
            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = (epoch * len(train_dataloader) + train_step + 1) * (rank + 1)
            # ---- Logging, save summery  ----
            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1

        # Terminal print for each epoch
        end_time = time.time()
        elapsed_time = end_time - start_time
        if is_main:
            print('=' * 20, 'System completed epoch: ', epoch+1, "/", args.max_epoch, ", Training time:", f"{elapsed_time:.2f}s", '=' * 20, flush=True)

        # ----- Save checkpoint ----- 
        # We save the model only from the first GPU process to avoid them overwriting each other. 
        dist.barrier()  # Synchronize all processes before saving

        # check overall elapsed time since training_start_time (11 hours 30 minutes)
        elapsed_since_start = time.time() - training_start_time
        time_limit_reached = elapsed_since_start >= (11.5 * 60 * 60)

        if is_main and ((epoch + 1) % args.ckpt_freq_epoch == 0 or (epoch + 1) == args.max_epoch or time_limit_reached):
            print("GPU ", rank, '=' * 20, " saving checkpoint...",'=' * 20, flush=True)
            to_save = {
                "model": pointpillars.module.state_dict()
                        if hasattr(pointpillars, "module") else pointpillars.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(to_save, os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        # ----- Validation ----- 
        if args.validation and epoch % 2 == 0:
            continue
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader, disable=not is_main)): # Only show progress bar on main process
                # move the tensors to the cuda
                for key in data_dict:
                            for j, item in enumerate(data_dict[key]):
                                if torch.is_tensor(item):
                                    data_dict[key][j] = item.to(rank, non_blocking=True)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
        	    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                # We acount for multiple GPUs when logging (rank)
                global_step = (epoch * len(val_dataloader) + val_step + 1) * (rank + 1)
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        pointpillars.train()

    if is_main:
        print("Training complete!", flush=True)
    cleanup() # Cleanup after use of multiple GPUs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/ceph/project/P7_Gravel_Gun/Datasets/Drone_Data', 
                        help='your data root for the drone dataset')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=4) # batch size per GPU
    parser.add_argument('--num_workers', type=int, default=5) # number of data loading cpu workers. 
    parser.add_argument('--nclasses', type=int, default=1) # number of classes to predict
    parser.add_argument('--init_lr', type=float, default=0.00025) 
    parser.add_argument('--max_epoch', type=int, default=160) 
    parser.add_argument('--log_freq', type=int, default=8) # How often we log to tensorboard (in steps)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=5)  # How often we save checkpoints (in epochs)
    parser.add_argument('--world_size', type=int, default=1) # number of gpus for training
    parser.add_argument('--pretrained', default='None')
    parser.add_argument('--validation', action='store_true', help='whether to run validation')
    parser.add_argument('--model', type=str, default='simple', help='model type')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    # Multiprocessing for multiple GPUs
    mp.spawn(main_workers, args=(args.world_size, args), nprocs=args.world_size, join=True)
