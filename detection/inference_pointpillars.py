'''
@Time       : 
@Author     : Jingsen Zheng
@File       : inference_pointpillars
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
from pointnet2.models.pointpillars_sem import PointPillarsSem

import torch

from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from detection.waymo_dataset_loader import WaymoDatasetLoader
from detection.waymo_dataset_loader import DatasetLoader

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-num_points",
    type=int,
    default=65536,
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-checkpoint", type=str,
    # default="/home/jingsen/workspace/Pointnet2_PyTorch/checkpoints/pointnet2_semseg_best.pth.tar", help="Checkpoint to start from"
    default="/home/jingsen/workspace/Pointnet2_PyTorch/checkpoints/pointpillars_best_.pth.tar", help="Checkpoint to start from"
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
    default="/media/jingsen/zjs_ssd/Dataset/waymo/extract_data_bbox/label_points/segment-16102220208346880_1420_000_1440_000_with_camera_labels/test",
    help="test data path"
)
parser.add_argument(
    "-train_data_path",
    type=str,
    default="/media/jingsen/zjs_ssd/Dataset/waymo/extract_data_bbox/label_points/segment-16102220208346880_1420_000_1440_000_with_camera_labels/train",
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

def show_points_with_colors(points, colors):
    point_cloud = open3d.PointCloud()
    point_cloud.points = open3d.Vector3dVector(np.array(points))
    point_cloud.colors = open3d.Vector3dVector(np.array(colors))
    visualization.custom_draw_geometry_with_key_callback(point_cloud)

if __name__ == "__main__":
    args = parser.parse_args()

    test_loader = DatasetLoader(args.test_data_path, 1, training=False, augment=True)

    model = PointPillarsSem(num_classes=2, input_channels=0, use_xyz=True)
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
        batch_ids, points, labels, origin_points, batch_size = data

        # show_two_label_points(points.numpy(), labels.numpy())

        points = points.to("cuda", non_blocking=True).transpose(dim0=0, dim1=1)
        labels = labels.to("cuda", non_blocking=True)
        batch_ids = batch_ids.to("cuda", non_blocking=True).int()

        preds = model(batch_ids, points, batch_size)
        _, classes = torch.max(preds, -1)

        classes = classes.cpu().numpy()
        labels = labels.cpu().numpy()
        colors = np.zeros([labels.shape[0], 3], dtype=int)
        # TP green
        idx = (labels == 1) & (classes == 1)
        colors[idx] = [0, 255, 0]
        # FP red
        idx = (classes ==1) & (labels == 0)
        colors[idx] = [255, 0, 0]
        # FN yellow
        idx = (classes == 0) & (labels == 1)
        colors[idx] = [255, 255, 0]
        # TN
        # default color

        # show_two_label_points(origin_points, classes.cpu().numpy())
        show_points_with_colors(origin_points, colors)
        print(classes.shape)

