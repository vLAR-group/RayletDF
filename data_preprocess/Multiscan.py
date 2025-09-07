import zlib
import os
import cv2
import shutil
from PIL import Image
import subprocess
import csv
import sys
import numpy as np
from tqdm import tqdm
import imageio as iio
import open3d as o3d
import glob
import json

import argparse

parser = argparse.ArgumentParser(description="Multiscan data preprocess")
parser.add_argument("--out_path", type=str, default="/media/HDD2/Multiscan/Multiscan_nerf")
parser.add_argument("--raw_data_path", type=str, default="/media/HDD2/Multiscan/raw_multiscan")
parser.add_argument("--split_path", type=str, default="/media/HDD2/Multiscan/dataset/benchmark/scans_split.csv")
args = parser.parse_args()

def run_command(cmd: str, verbose=False, exit_on_error=True):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr is not None:
            print(out.stderr.decode("utf-8"))
        if exit_on_error:
            sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out

split = ["test", "train"]
split_dict = {"train": [], "test":[]}
select_interval_eval = 10
level = 2 # Control the depth level, depth confidence < level will be set to 0
with open(args.split_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if row[1] == "train":
            split_dict['train'].append(row[0])
        if row[1] == "test":
            split_dict['test'].append(row[0])
raw_data_dir = args.raw_data_path
dpt_height, dpt_width, sample_rate, depth_scale = 192, 256, 1, 1000
error_list = []
for s in split:
    out_dir = f'{args.out_path}/{s}'
    scene_list = split_dict[s]
    scene_list.sort()
    scale_sta = []
    for scene_name in tqdm(scene_list):
        pose_info, K_info = [], []
        try:
            scene_path = os.path.join(raw_data_dir, scene_name)
            with open(os.path.join(scene_path, f"{scene_name}.jsonl"), "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    pose_info.append(json_obj["transform"])
                    K_info.append(json_obj["intrinsics"])
            with open(os.path.join(scene_path, f"{scene_name}.align.json"), "r") as f:
                align_info = json.load(f)
            align_info = align_info["coordinate_transform"]
            align_info = np.array(align_info).reshape(4, 4).transpose()
            align_info= align_info / align_info[3][3]
            # align_info = np.linalg.inv(align_info)
            rgb_path = os.path.join(scene_path, "rgb")
            os.makedirs(rgb_path, exist_ok=True)
            ### cmd = f'ffmpeg -i {os.path.join(scene_path, scene_name+".mp4")} -vf fps=10 -start_number 0 -q:v 1 {rgb_path}/frame_%06d.jpg'
            cmd = f'ffmpeg -i {os.path.join(scene_path, scene_name+".mp4")} -start_number 0 -q:v 1 {rgb_path}/frame_%06d.jpg'
            run_command(cmd, verbose=True)
            depth_path = os.path.join(scene_path, "depth")
            os.makedirs(depth_path, exist_ok=True)
            with open(os.path.join(scene_path, scene_name+".depth.zlib"), 'rb') as infile:
                data = infile.read()
                data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(data, dtype=np.float16).reshape(-1, dpt_height, dpt_width).copy()

            with open(os.path.join(scene_path, scene_name+".confidence.zlib"), 'rb') as infile:
                data = infile.read()
                data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                confidence = np.frombuffer(data, dtype=np.uint8).reshape(-1, dpt_height, dpt_width).copy()
            depth[confidence < level] = 0
            for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc='decode_depth'):
                iio.imwrite(f"{depth_path}/frame_{frame_id:06}.png", (depth[frame_id] * 1000).astype(np.uint16))

            trans_info = {'train': {"depth_scale": depth_scale, 'frames': []},'test': {"depth_scale": depth_scale, 'frames': []},'all': { "depth_scale": depth_scale, 'frames': []},}
            mesh_path = os.path.join(scene_path, f"{scene_name}.ply")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.transform(align_info)
            # aabb = mesh.get_axis_aligned_bounding_box()
            # min_bound = aabb.get_min_bound()
            # max_bound = aabb.get_max_bound()
            # dimensions = max_bound - min_bound
            # scale_sta.append(dimensions[0]*dimensions[1]*dimensions[2])
            sample_points_num = int(mesh.get_surface_area()) * 50
            if sample_points_num<250:
                continue
            pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_points_num)
            out_rgb_dir = os.path.join(out_dir, scene_name, 'input')
            out_depth_dir = os.path.join(out_dir, scene_name, 'depth')
            out_highres_dir = os.path.join(out_dir, scene_name, 'highres')
            os.makedirs(out_rgb_dir, exist_ok=True)
            os.makedirs(out_depth_dir, exist_ok=True)
            os.makedirs(out_highres_dir, exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(out_dir, scene_name, 'points3d.ply'), pcd)
            o3d.io.write_triangle_mesh(os.path.join(out_dir, scene_name, scene_name+'_mesh.ply'), mesh)
            depth_images = os.listdir(depth_path)
            depth_images.sort()
            select_image_idx = [idx for idx, _ in enumerate(depth_images) if idx % 50 == 0]
            for i, idx in enumerate(select_image_idx):
                pose, K = pose_info[idx], K_info[idx]
                if np.isinf(pose).any():
                    print(f'inf pose: {scene_name}')
                    continue
                pose = np.array(pose)
                pose = pose.reshape(4, 4).transpose()
                pose= pose / pose[3][3]
                depth_name = depth_images[idx]
                depth = os.path.join(depth_path, depth_name)
                rgb_name = depth_name.replace("png", "jpg")
                rgb = os.path.join(rgb_path, rgb_name)
                fx, fy, cx, cy = K[0], K[4], K[6], K[7]
                fovx = 2 * np.arctan(cx / fx)
                info = {'file_path': f'input/{rgb_name}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{depth_name}', 
                        'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, 'highres_file_path': f'highres/{rgb_name}'}
                if (i+1) % select_interval_eval ==0:
                    trans_info['test']['frames'].append(info)
                else:
                    trans_info['train']['frames'].append(info)
                trans_info["all"]['frames'].append(info)
            
            for split in ['train', 'test', 'all']:
                savedir = os.path.join(out_dir, scene_name)
                with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                    json.dump(trans_info[split], fp, indent=4)
        except Exception as e:
            print(e)
            print(f'error: {scene_name}')
            error_list.append(scene_name)

    # print(np.mean(scale_sta), np.max(scale_sta), np.min(scale_sta))
    print(f'error_list: {error_list}')


