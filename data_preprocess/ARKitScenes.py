import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
import glob
import argparse

parser = argparse.ArgumentParser(description="ARKitScenes data preprocess")
parser.add_argument("--out_path", type=str, default="/media/HDD2/ARKitScenes/ARKitScenes_nerf")
parser.add_argument("--raw_data_path", type=str, default="/media/HDD2/ARKitScenes/raw_ARKitScenes")
args = parser.parse_args()

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return fx, fy, hw, hh
    # return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt
scale_sta = []
split = ['train', 'test']
for s in split:
    raw_data_dir = f'{args.raw_data_path}/3dod/{s}'  # path to raw data
    scene_list = os.listdir(raw_data_dir)
    scene_list.sort()
    select_interval_eval = 10
    out_dir = f'{args.out_path}/{s}'
    # FOVX = np.arctan(319.5 / 577.870605) * 2
    for scene in tqdm(scene_list):
        traj_file = os.path.join(raw_data_dir, scene, scene+"_frames", 'lowres_wide.traj')
        with open(traj_file) as f:
            traj = f.readlines()
        # convert traj to json list
        poses_from_traj = {}
        for line in traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)

        trans_info = {'train': {"depth_scale": 1000.0, 'frames': []},
                'test': {"depth_scale": 1000.0, 'frames': []},
                'all': { "depth_scale": 1000.0, 'frames': []},}
        scene_dir = os.path.join(raw_data_dir, scene, scene+"_frames",)
        rgb_dir = os.path.join(scene_dir, 'lowres_wide')
        depth_dir = os.path.join(scene_dir, 'lowres_depth')

        depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        frame_ids = [os.path.basename(x) for x in depth_images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids = [x for x in frame_ids]
        frame_ids.sort()
        intrinsics = {}
        select_frame_ids = [c for idx, c in enumerate(frame_ids) if idx % 5 == 0]
        if os.path.exists(os.path.join(out_dir, scene)):
            mesh_path = os.path.join(raw_data_dir, scene, scene+"_3dod_mesh.ply")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            aabb = mesh.get_axis_aligned_bounding_box()
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()
            dimensions = max_bound - min_bound
            scale_sta.append(dimensions[0]*dimensions[1]*dimensions[2])
        else:
            # color_img_list = os.listdir(rgb_dir)
            # color_img_list.sort(key=lambda x:int(x.split('.')[0]))
            out_rgb_dir = os.path.join(out_dir, scene, 'input')
            out_depth_dir = os.path.join(out_dir, scene, 'depth')
            os.makedirs(out_rgb_dir, exist_ok=True)
            os.makedirs(out_depth_dir, exist_ok=True)
            # mesh_path = os.path.join(scene_dir, scene+"_vh_clean_2.ply")
            mesh_path = os.path.join(raw_data_dir, scene, scene+"_3dod_mesh.ply")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            sample_points_num = int(mesh.get_surface_area()) * 100
            if sample_points_num<500:
                continue
            pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_points_num)
            o3d.io.write_point_cloud(os.path.join(out_dir, scene, 'points3d.ply'), pcd)
            shutil.copy2(mesh_path, os.path.join(out_dir, scene, scene+'_mesh.ply'))


            aabb = mesh.get_axis_aligned_bounding_box()
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()
            dimensions = max_bound - min_bound
            scale_sta.append(dimensions[0]*dimensions[1]*dimensions[2])
            # o3d.io.write_triangle_mesh(os.path.join(out_dir, scene, 'mesh.ply'), mesh)
            for idx, frame_id in enumerate(select_frame_ids):
                if str(frame_id) in poses_from_traj.keys():
                    pose = np.array(poses_from_traj[str(frame_id)])
                else:
                    for my_key in list(poses_from_traj.keys()):
                        if abs(float(frame_id) - float(my_key)) < 0.005:
                            pose = poses_from_traj[str(my_key)]
                
                if np.isinf(pose).any():
                    print(f'inf pose: {frame_id}')
                    continue
                pose[:3, 1:3] *= -1
                img = "{}_{}.png".format(scene, frame_id)
                img_path = os.path.join(rgb_dir, img)
                # rgb = cv2.imread(img_path)
                depth_path = img_path.replace('lowres_wide', 'lowres_depth')
                # rgb = cv2.resize(rgb, depth_size)
                shutil.copy2(img_path, os.path.join(out_rgb_dir, img))
                # cv2.imwrite(os.path.join(out_rgb_dir, img), rgb)
                out_depth_path = os.path.join(out_depth_dir, img)
                shutil.copy2(depth_path, out_depth_path)
                dpt_name = img
                intrinsic_fn = os.path.join(scene_dir, "lowres_wide_intrinsics", f"{scene}_{frame_id}.pincam")
                fx, fy, cx, cy = st2_camera_intrinsics(intrinsic_fn)
                fovx = 2 * np.arctan(cx / fx)
                info = {'file_path': f'input/{img}', 'transform_matrix': pose.tolist(), "depth_file_path": f'depth/{dpt_name}', 
                        'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx}
                if (idx+1) % select_interval_eval ==0:
                    trans_info['test']['frames'].append(info)
                else:
                    trans_info['train']['frames'].append(info)
                trans_info["all"]['frames'].append(info)
            
            for split in ['train', 'test', 'all']:
                savedir = os.path.join(out_dir, scene)
                with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                    json.dump(trans_info[split], fp, indent=4)

print(np.mean(scale_sta), np.max(scale_sta), np.min(scale_sta))

