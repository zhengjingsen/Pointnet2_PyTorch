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
import torch
import torch.utils.data as data
import numpy as np
import os
import subprocess
import shlex

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
        pt_idxs = np.random.choice(self.points[idx].shape[0], self.num_points, replace=False)

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
