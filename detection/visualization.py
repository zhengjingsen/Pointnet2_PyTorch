import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def custom_draw_geometry_with_key_callback(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    def load_render_option(vis):
        vis.get_render_option().load_from_json(
                "../../TestData/renderoption.json")
        return False
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.draw_geometries_with_key_callbacks([pcd], key_to_callback)
#
# if __name__ == "__main__":
#
#
#     points = np.random.rand(10000, 3)
#     # print("Load a ply point cloud, print it, and render it")
#     if os.path.exists(args.pointcloud_file):
#         points = o3d.io.read_point_cloud(args.pointcloud_file)
#     # o3d.visualization.draw_geometries([pcd])
#
#     # line_set = o3d.geometry.LineSet()
#     # line_set.points = o3d.utility.Vector3dVector(points)
#     # line_set.lines = o3d.utility.Vector2iVector(lines)
#     # line_set.colors = o3d.utility.Vector3dVector(colors)
#     # colors = [np.sin(points[:, 0]), np.sin(points[:, 1]), np.sin(points[:, 2])]
#     # colors = np.array(colors)
#
#     # point_cloud = o3d.PointCloud()
#     # point_cloud.points = o3d.Vector3dVector(points)
#     # point_cloud.colors = o3d.Vector3dVector(colors.transpose())
#     # o3d.visualization.draw_geometries([point_cloud])
#     custom_draw_geometry_with_key_callback(points)