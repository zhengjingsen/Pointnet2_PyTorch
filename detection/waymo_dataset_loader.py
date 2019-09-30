'''
@Time       : 
@Author     : Jingsen Zheng
@File       : waymo_dataset_loader
@Brief      : 
'''

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import math
import torch
import torch.utils.data as data
import numpy as np
import os
import subprocess
import shlex
from scipy.spatial.transform import Rotation

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = np.fromfile(name)
    f = f.reshape(-1, 4)
    data = f[:, :3]
    label = f[:, 3].astype(np.int64)
    # label = np.expand_dims(label, axis=1)
    return data, label


class WaymoDatasetLoader(data.Dataset):
    def __init__(self, data_path, num_points):
        super(WaymoDatasetLoader, self).__init__()

        self.data_dir = data_path
        self.num_points = num_points

        all_files = os.listdir(data_path)

        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = _load_data_file(os.path.join(data_path, f))
            data_batchlist.append(data)
            label_batchlist.append(label)

        self.points = data_batchlist
        self.labels = label_batchlist

    def __getitem__(self, idx):
        if self.num_points is not None:
            pt_idxs = np.random.choice(self.points[idx].shape[0], self.num_points, replace=False)
        else:
            pt_idxs = np.arange(self.points[idx].shape[0])

        current_points = torch.from_numpy(self.points[idx][pt_idxs, :].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx][pt_idxs].copy()).type(
            torch.LongTensor
        )

        return current_points, current_labels

    def __len__(self):
        return int(len(self.points))

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


class DatasetLoader:
    def __init__(self, data_path, batch_size, num_points = None, augment = False,
                 min_range_x = -75, max_range_x = 75,
                 min_range_y = -75, max_range_y = 75,
                 min_range_z = -1,  max_range_z = 5,
                 max_shift_x = 0.1, max_shift_y = 0.1,
                 max_shift_z = 0.1, max_rotation = 1):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_points = num_points
        self.augment = augment

        self.min_range_x = min_range_x
        self.max_range_x = max_range_x
        self.min_range_y = min_range_y
        self.max_range_y = max_range_y
        self.min_range_z = min_range_z
        self.max_range_z = max_range_z

        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.max_shift_z = max_shift_z
        self.max_rotation = max_rotation

        all_files = os.listdir(data_path)
        self.data_size = len(all_files)

        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = self.load_pointcloud(os.path.join(data_path, f))
            data_batchlist.append(data)
            label_batchlist.append(label)

        self.points = data_batchlist
        self.labels = label_batchlist

    def data_augment(self, points, labels):
        # 1) No ring information. Hence skipping ring based augmentation

        # 2) Add random noise [-rnd, rnd]
        # points[:, 3] += 2 * self.cfg.RAND_INTENSITY_VARIATION * np.random.rand() - self.cfg.RAND_INTENSITY_VARIATION
        # 3) Clamp the intensity between [0, 1]
        # points[:, 3] = np.clip(points[:, 3], 0.0, 1.0)

        # 4) Shift augment along X and Y - direction
        shift_x = 2 * self.max_shift_x * np.random.rand() - self.max_shift_x
        shift_y = 2 * self.max_shift_y * np.random.rand() - self.max_shift_y

        points[:, 0] += shift_x
        points[:, 1] += shift_y

        # 5) Augment along z-axis
        move_z = 2 * self.max_shift_z * np.random.rand() - self.max_shift_z
        points[:, 2] += move_z

        # 4) Generate a random rotation angle [-rnd_rot, rnd_rot]
        rand_rot = np.random.randint(2 * self.max_rotation + 1) - self.max_rotation
        # 5) Find the rotation matrix and apply that rotation
        rotMat = Rotation.from_euler('z', rand_rot, degrees=True)
        rot_res = rotMat.apply(points[:, :3])
        points[:, :3] = rot_res

        # delete points outside range
        req_mask = (points[:, 0] > self.min_range_x) & (points[:, 0] < self.max_range_x) & (
                    points[:, 1] > self.min_range_y) & (points[:, 1] < self.max_range_y) & (
                    points[:, 2] > self.min_range_z) & (points[:, 2] < self.max_range_z)
        points = points[req_mask]
        label = labels[req_mask]

        # normalize the x and y of scan to lie in [-1, 1]
        points[:, 0] = points[:, 0] / self.max_range_x
        points[:, 1] = points[:, 1] / self.max_range_y
        points[:, 2] = points[:, 2] / self.max_range_z

        # Randomly shuffle the points & labels around to remove ordering dependance
        p = np.random.permutation(len(points))
        points = points[p]
        labels = labels[p]

        return points, labels

    def load_pointcloud(self, name):
        f = np.fromfile(name)
        f = f.reshape(-1, 4)
        data = f[:, :3]
        label = f[:, 3].astype(np.int64)

        # center along z-axis first
        mean_z = np.mean(data[:, 2])
        data[:, 2] = data[:, 2] - mean_z

        return data, label

    def __iter__(self):
        self.cur_it = 0
        self.index = np.arange(self.data_size)
        np.random.shuffle(self.index)
        return self

    def __next__(self):
        if self.cur_it >= 0:
            batch_ids = []
            points_batch = []
            labels_batch = []

            for i in range(self.batch_size):
                if self.cur_it < self.data_size:
                    idx = self.index[self.cur_it]
                else:
                    idx = self.index[np.random.randint(0, self.data_size - 1)]

                if self.num_points is not None:
                    pt_idxs = np.random.choice(self.points[idx].shape[0], self.num_points, replace=False)
                else:
                    pt_idxs = np.arange(self.points[idx].shape[0])

                points = self.points[idx][pt_idxs, :].copy()
                labels = self.labels[idx][pt_idxs].copy()

                if self.augment:
                    points, labels = self.data_augment(points, labels)

                batch_ids.append(np.repeat(i, len(points)))
                points_batch.append(points)
                labels_batch.append(labels)

                self.cur_it = self.cur_it + 1

            if self.cur_it >= self.data_size:
                self.cur_it = -1
            return torch.from_numpy(np.hstack(batch_ids)), \
                   torch.from_numpy(np.concatenate(points_batch, axis=0)).type(torch.FloatTensor), \
                   torch.from_numpy(np.concatenate(labels_batch, axis=0)).type(torch.LongTensor), \
                   self.batch_size
        else:
            raise StopIteration

    def iter_num_per_epoch(self):
        return math.ceil(self.data_size / self.batch_size)