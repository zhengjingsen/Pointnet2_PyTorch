'''
@Time       : 
@Author     : Jingsen Zheng
@File       : test
@Brief      : 
'''

import h5py
import argparse
# from detection import visualization
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file_name', default='', help='')

args = parser.parse_args()

# file = h5py.File(args.h5_file_name)
# data = file['data'][:]
# label = file['label'][:]
#
# points = data[:10, :, :3]
# points = np.reshape(points, (-1, 3))
# colors = data[:10, :, 3:6]
# colors = np.reshape(colors, (-1, 3))
#
# point_cloud = open3d.PointCloud()
# point_cloud.points = open3d.Vector3dVector(np.array(points))
# point_cloud.colors = open3d.Vector3dVector(np.array(colors))
# # o3d.visualization.draw_geometries([point_cloud])
# visualization.custom_draw_geometry_with_key_callback(point_cloud)

# x = torch.randn(3)
# print(x)
#
# y = x.repeat_interleave(torch.tensor([2, 3, 4]), dim=0)
# print(y)
#
# z = torch.linspace(0, 10, 11).int()
# print(z)

class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            return -1
            raise StopIteration

for num in iter(MyNumbers()):
    print(num)
