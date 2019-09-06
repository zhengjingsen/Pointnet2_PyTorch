'''
@Time       : 
@Author     : Jingsen Zheng
@File       : test
@Brief      : 
'''

import h5py
import argparse
import open3d
from detection import visualization
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file_name', default='', help='')

args = parser.parse_args()

file = h5py.File(args.h5_file_name)
data = file['data'][:]
label = file['label'][:]

points = data[:10, :, :3]
points = np.reshape(points, (-1, 3))
colors = data[:10, :, 3:6]
colors = np.reshape(colors, (-1, 3))

point_cloud = open3d.PointCloud()
point_cloud.points = open3d.Vector3dVector(np.array(points))
point_cloud.colors = open3d.Vector3dVector(np.array(colors))
# o3d.visualization.draw_geometries([point_cloud])
visualization.custom_draw_geometry_with_key_callback(point_cloud)