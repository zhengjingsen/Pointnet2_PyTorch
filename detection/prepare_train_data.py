import os
import argparse
import visualization
import open3d
import numpy as np
import torch
import pointnet2.utils.pointnet2_utils as pointnet_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path',
                    default='/home/jingsen/laptop/225A6D42D4FA828F/waymo',
                    # '/media/jingsen/3aa94184-0fba-45c2-8122-335db3d9776b/Dataset/waymo/extract_data',
                    help='path to extracted waymo dataset')
parser.add_argument('--segment_name',
                    default='segment-16102220208346880_1420_000_1440_000_with_camera_labels',
                    # 'segment-16102220208346880_1420_000_1440_000_with_camera_labels',
                    help='segment of dataset')
parser.add_argument('--radius', type=float, default=1.0, help='radius to label points')
parser.add_argument('--nsample', type=int, default=300, help='max sample num for each object')
parser.add_argument('--save_path',
                    default='/media/jingsen/3aa94184-0fba-45c2-8122-335db3d9776b/Dataset/waymo/extract_data',
                    help='path to save label points')

args = parser.parse_args()

def load_label_file(label_file):
    labels = np.loadtxt(label_file, dtype=np.str)
    return labels

def extract_label_points(pcd_file, label_file):
    points = np.asarray(open3d.io.read_point_cloud(pcd_file).points)
    colors = np.zeros(points.shape, dtype=int)
    points_label = np.zeros(points.shape[0], dtype=np.uint8)

    xyz = torch.from_numpy(np.expand_dims(points, axis=0)).type(torch.FloatTensor)
    xyz = xyz.to("cuda", non_blocking=True)
    labels = load_label_file(label_file)

    for label in labels:
        if (int(label[0]) == 1):            # vehicle
            center = np.array([float(label[2]), float(label[3]), float(label[7]) * 0.5]) #float(label[4])])
            size = np.array([float(label[5]), float(label[6]), float(label[7])])
            heading = float(label[8])

            center_shift = np.array([size[0] * np.cos(heading) * 0.5, size[0] * np.sin(heading) * 0.5, 0])
            proposal_centers = np.asarray([[center + center_shift, center - center_shift]])

            idx = pointnet_utils.ball_query(size[2] * 0.5, args.nsample,
                                            xyz, torch.from_numpy(proposal_centers).type(torch.Tensor).to("cuda", non_blocking=True))
            idx = np.squeeze(idx.cpu().numpy())

            idx1 = np.unique(idx[0])
            idx2 = np.unique(idx[1])

            if idx1.size > idx2.size:
                # colors[idx1] = [255, 0, 0]
                points_label[idx1] = 1
            else:
                # colors[idx2] = [255, 0, 0]
                points_label[idx2] = 1

    return points, colors, points_label

def show_color_points(points, colors):
    point_cloud = open3d.PointCloud()
    point_cloud.points = open3d.Vector3dVector(np.array(points))
    point_cloud.colors = open3d.Vector3dVector(np.array(colors))

    visualization.custom_draw_geometry_with_key_callback(point_cloud)

def main():
    if not os.path.exists(args.dataset_path):
        print('dateset_path does not exists!')

    pcd_path = args.dataset_path + '/pointcloud/' + args.segment_name
    label_path = args.dataset_path + '/labels/' + args.segment_name
    bin_path = args.save_path + '/label_points/' + args.segment_name
    pcd_list = os.listdir(pcd_path)

    if not os.path.exists(bin_path):
        os.makedirs(bin_path)

    for pcd in pcd_list:
        file_prefix = pcd.split('.')[0]
        pcd_file = pcd_path + '/' + file_prefix + '.pcd'
        label_file = label_path + '/' + file_prefix + '.txt'
        points, colors, points_label = extract_label_points(pcd_file, label_file)
        points_label = np.expand_dims(points_label, axis=1)
        points_with_label = np.concatenate((points, points_label), axis=1)
        points_with_label.tofile(bin_path + '/' + file_prefix + '.bin')
        # show_color_points(points, colors)


if __name__ == '__main__':
    main()