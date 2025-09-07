import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json

import argparse

parser = argparse.ArgumentParser(description="Scannet data preprocess")
parser.add_argument("--out_path", type=str, default="/media/HDD2/scannet_nerf/scans")
parser.add_argument("--raw_data_path", type=str, default="/media/HDD2/scannet_raw/scans", help="Path to the Scannet raw data (RGB, Depth and Pose), extracted from .sens")
parser.add_argument("--v2_path", type=str, default="/media/HDD2/ScanNetv2/scans", help="Path to the ScannetV2 mesh")
args = parser.parse_args()

v2dir = args.v2_path  # Path to the ScannetV2 mesh
raw_data_dir = args.raw_data_path  # Path to the Scannet raw data (RGB, Depth and Pose)
scene_list = os.listdir(raw_data_dir)
scene_list.sort()
select_interval_eval = 10
depth_size = (640, 480)
out_dir = args.out_path
FOVX = np.arctan(319.5 / 577.870605) * 2
for scene in tqdm(scene_list):
    trans_info = {'train': {'fx': 577.870605, 'fy': 577.870605, 'cx':319.5, 'cy':239.5, "camera_angle_x":FOVX, "depth_scale": 1000.0, 'frames': []},
              'test': {'fx': 577.870605, 'fy': 577.870605, 'cx':319.5, 'cy':239.5, "camera_angle_x":FOVX, "depth_scale": 1000.0, 'frames': []},
              'all': {'fx': 577.870605, 'fy': 577.870605, 'cx':319.5, 'cy':239.5, "camera_angle_x":FOVX, "depth_scale": 1000.0, 'frames': []},}
    scene_dir = os.path.join(raw_data_dir, scene)
    rgb_dir = os.path.join(scene_dir, 'color')
    depth_dir = os.path.join(scene_dir, 'depth')
    color_img_list = os.listdir(rgb_dir)
    color_img_list.sort(key=lambda x:int(x.split('.')[0]))
    select_color = [c for idx, c in enumerate(color_img_list) if idx % 5 == 0]
    out_rgb_dir = os.path.join(out_dir, scene, 'input')
    out_depth_dir = os.path.join(out_dir, scene, 'depth')
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    # mesh_path = os.path.join(scene_dir, scene+"_vh_clean_2.ply")
    mesh_path = os.path.join(v2dir, scene, scene+"_vh_clean_2.ply")
    shutil.copy2(mesh_path, os.path.join(out_dir, scene, scene+'_mesh.ply'))
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    sample_points_num = int(mesh.get_surface_area()) * 150
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_points_num)
    o3d.io.write_point_cloud(os.path.join(out_dir, scene, 'points3d.ply'), pcd)
    # o3d.io.write_triangle_mesh(os.path.join(out_dir, scene, 'mesh.ply'), mesh)
    for idx, img in enumerate(select_color):
        img_path = os.path.join(rgb_dir, img)
        rgb = cv2.imread(img_path)
        depth_path = img_path.replace('color', 'depth').replace('jpg', 'png')
        rgb = cv2.resize(rgb, depth_size)
        cv2.imwrite(os.path.join(out_rgb_dir, img), rgb)
        out_depth_path = os.path.join(out_depth_dir, img.replace('jpg', 'png'))
        shutil.copy2(depth_path, out_depth_path)
        pose_path = os.path.join(scene_dir, 'pose', img.replace('jpg', 'txt'))
        pose = np.loadtxt(pose_path)
        pose[:3, 1:3] *= -1
        if np.isinf(pose).any():
            print(f'inf pose: {pose_path}')
            continue
        dpt_name = img.replace('jpg', 'png')
        info = {'file_path': f'input/{img}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{dpt_name}'}
        if (idx+1) % select_interval_eval ==0:
            trans_info['test']['frames'].append(info)
        else:
            trans_info['train']['frames'].append(info)
        trans_info["all"]['frames'].append(info)
    
    for split in ['train', 'test', 'all']:
        savedir = os.path.join(out_dir, scene)
        with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
            json.dump(trans_info[split], fp, indent=4)

