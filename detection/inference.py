'''
@Time       : 
@Author     : Jingsen Zheng
@File       : inference
@Brief      : 
'''
import os
import argparse
import visualization
import open3d
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
import etw_pytorch_utils as pt_utils
import argparse


import torch

from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from detection.waymo_dataset_loader import WaymoDatasetLoader

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-batch_size", type=int, default=1, help="Batch size [default: 32]"
)
parser.add_argument(
    "-num_points",
    type=int,
    default=65536,
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-checkpoint", type=str, default="/home/jingsen/workspace/Pointnet2_PyTorch/checkpoints/pointnet2_semseg_best.pth.tar", help="Checkpoint to start from"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger",
)
parser.add_argument(
    "-test_data_path",
    type=str,
    default="/media/jingsen/3aa94184-0fba-45c2-8122-335db3d9776b/Dataset/waymo/extract_data_bbox/label_points/segment-16102220208346880_1420_000_1440_000_with_camera_labels/test",
    help="test data path"
)
parser.add_argument(
    "-train_data_path",
    type=str,
    default="/media/jingsen/3aa94184-0fba-45c2-8122-335db3d9776b/Dataset/waymo/extract_data_bbox/label_points/segment-16102220208346880_1420_000_1440_000_with_camera_labels/train",
    help="train data path"
)

def show_two_label_points(points, labels):
    point_cloud = open3d.PointCloud()
    point_cloud.points = open3d.Vector3dVector(np.array(points))

    colors = np.zeros([labels.shape[0], 3], dtype=int)
    idx = labels == 1
    colors[idx] = [255, 0, 0]
    # for i in range(colors.shape[0]):
    #     if labels[i] == 1:
    #         colors[i] = [255, 0, 0]
    point_cloud.colors = open3d.Vector3dVector(np.array(colors))

    visualization.custom_draw_geometry_with_key_callback(point_cloud)


if __name__ == "__main__":
    args = parser.parse_args()

    test_set = WaymoDatasetLoader(args.train_data_path, None)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    model = Pointnet(num_classes=2, input_channels=0, use_xyz=True)
    model.cuda()

    print("checkpoint filename: {}".format(args.checkpoint.split(".")[0]))

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, None, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    data_iter = iter(test_loader.__iter__())
    for data in data_iter:
        inputs, labels = data

        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        preds = model(inputs)
        _, classes = torch.max(preds, -1)
        show_two_label_points(inputs.cpu().numpy()[0, ...], classes.cpu().numpy()[0])
        print(classes.shape)

