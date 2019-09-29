from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
from collections import namedtuple
from torch_scatter import scatter_max
import pointnet2.models.config as config

class conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(conv1d, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            )
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, 1)
        
    def forward(self, x):
        x = self.conv(torch.unsqueeze(x, 0))
        return x[0]

class vfe(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(vfe, self).__init__()
        self.conv = conv1d(in_ch, int(out_ch/2))

    def forward(self, x, group_ids, grid_num):
        out0 = self.conv(x)
        s_out, s_max = scatter_max(out0, group_ids, dim=1, dim_size=grid_num, fill_value=0)
        out1 = s_out[:, group_ids]
        output = torch.cat((out0, out1), 0)

        return output

class conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, padding=1):
        super(conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)

        return x

class interpolate(nn.Module):
    def __init__(self, channel, min_x, min_y, x_num, y_num, grid_size):
        super(interpolate, self).__init__()
        self.channel = channel
        self.min_x = min_x
        self.min_y = min_y
        self.x_num = x_num
        self.y_num = y_num
        self.grid_size = grid_size
    
    def forward(self, x, batch_size, batch_ids, xs, ys):
        xxs = (xs - self.min_x) / self.grid_size - 0.5
        yys = (ys - self.min_y) / self.grid_size - 0.5 + (batch_ids * self.y_num).float()
        #grid_sample requires coords (-1, 1)
        xxs_norm = 2 * xxs / self.x_num - 1
        yys_norm = 2 * yys / self.y_num / batch_size - 1
        grid = torch.unsqueeze(torch.unsqueeze(torch.cat((torch.unsqueeze(xxs_norm, -1), torch.unsqueeze(yys_norm, -1)), 1), 0), 0)

        x = x.permute(1, 0, 2, 3).reshape(1, self.channel, self.y_num*batch_size, self.x_num)
        x = torch.nn.functional.grid_sample(x, grid).reshape(self.channel, -1)

        return x

class PointPillarsSem(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True):
        super(PointPillarsSem, self).__init__()
        self.cfg = config.cfg
        self.x_num = config.get_x_num(self.cfg)
        self.y_num = config.get_y_num(self.cfg)

        self.conv1d_0 = conv1d(input_channels + 3, 16)
        self.conv1d_1 = conv1d(16, 16)

        self.vfe_0 = vfe(16 + 2, 32)
        self.vfe_1 = vfe(32, 32)

        self.conv1d_2 = conv1d(32, 32)

        self.conv2d_0 = conv2d(32, 32, kernel_size=3, dilation=1, padding=1)
        self.conv2d_1 = conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
        self.conv2d_2 = conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
        self.conv2d_3 = conv2d(32, 32, kernel_size=3, dilation=2, padding=2)

        self.interpolate = interpolate(32, self.cfg.MIN_X, self.cfg.MIN_Y, self.x_num, self.y_num, self.cfg.GRID_SIZE)

        self.conv1d_3 = conv1d(64, num_classes, activation=False)

    def forward(self, batch_ids, points, batch_size):
        cfg = self.cfg
        x_groups = ((points[0, :] - cfg.MIN_X) / cfg.GRID_SIZE).floor().int()
        y_groups = ((points[1, :] - cfg.MIN_Y) / cfg.GRID_SIZE).floor().int()
        # scatter_max needs int64 indices
        group_ids = (batch_ids * self.x_num * self.y_num + y_groups * self.x_num + x_groups).long()
        grid_num = batch_size * self.y_num * self.x_num
        
        # Begin of Pointwise layers
        x = self.conv1d_0(points)
        x = self.conv1d_1(x)

        voxel_xs = (x_groups.float() + 0.5) * cfg.GRID_SIZE + cfg.MIN_X
        voxel_ys = (y_groups.float() + 0.5) * cfg.GRID_SIZE + cfg.MIN_Y
        dx = torch.unsqueeze(points[0, :] - voxel_xs, 0)
        dy = torch.unsqueeze(points[1, :] - voxel_ys, 0)
        x = torch.cat((x, dx, dy), 0)

        x = self.vfe_0(x, group_ids, grid_num)
        x = self.vfe_1(x, group_ids, grid_num)

        point_0 = self.conv1d_2(x)
        # End of pointwise layers

        # Begin of Voxelwise layers
        x, argmax = scatter_max(point_0, group_ids, dim=1, dim_size=grid_num, fill_value=0)
        x = x.view(32, batch_size, self.y_num, self.x_num)
        x = x.permute(1, 0, 2, 3)

        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)

        point_1 = self.interpolate(x, batch_size, batch_ids, points[0, :], points[1, :])
        point = torch.cat((point_0, point_1), 0)

        score = self.conv1d_3(point)

        return score

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            batch_ids, points, labels, batch_size = data

            points = points.to("cuda", non_blocking=True).transpose(dim0=0, dim1=1)
            labels = labels.to("cuda", non_blocking=True)
            batch_ids = batch_ids.to("cuda", non_blocking=True).int()

            preds = model(batch_ids, points, batch_size)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

        return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn

if __name__ == "__main__":
    net = PointPillarsSem(2, 1)
    points = 2 * torch.rand(4, 8) - 1       # for pytorch: data order is NCHW
    batch_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).int()
    batch_size = 2
    seg = net.forward(batch_ids, points, batch_size)