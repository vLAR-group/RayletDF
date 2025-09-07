import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
import collections
from enum import Enum
from pathlib import Path
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
from plyfile import PlyData, PlyElement

import argparse

parser = argparse.ArgumentParser(description="Scannet++ data preprocess")
parser.add_argument("--out_path", type=str, default="/media/HDD2/Scannetpp/Scannetpp_nerf")
parser.add_argument("--raw_data_path", type=str, default="/media/HDD2/Scannetpp/data")
parser.add_argument("--split_path", type=str, default="/media/HDD2/Scannetpp/splits")
args = parser.parse_args()


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"

class ColmapImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = os.path.split(elems[9])[1]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = ColmapImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def parse_colmap_camera_params(camera):
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out

scale_sta = []
split = ['test', "train"]
for s in split:
    split_list_txt = f'{args.split_path}/nvs_sem_{s}.txt'.replace("test", "val")
    raw_data_dir = f'{args.raw_data_path}'  # path to raw Scannetpp data
    with open(split_list_txt) as f:
        scene_list = f.readlines()
    select_interval_eval = 10
    out_dir = f'{args.out_path}/{s}'
    for scene in tqdm(scene_list):
        scene = scene.strip()
        scene_path = os.path.join(raw_data_dir, scene)
        # if os.path.exists(os.path.join(out_dir, scene, scene+'_mesh.ply')):
        #     continue

        colmap_path = os.path.join(scene_path, "iphone", "colmap")
        rgb_dir = os.path.join(scene_path, "iphone", "rgb")
        mask_dir = os.path.join(scene_path, "iphone", "rgb_masks")
        depth_dir = os.path.join(scene_path, "iphone", "depth")

        if not os.listdir(depth_dir):
            print(f"scene {scene} has no depth")
            continue

        out_rgb_dir = os.path.join(out_dir, scene, 'input')
        out_depth_dir = os.path.join(out_dir, scene, 'depth')
        out_mask_dir = os.path.join(out_dir, scene, 'masks')
        out_highres_dir = os.path.join(out_dir, scene, 'highres')
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        os.makedirs(out_highres_dir, exist_ok=True)

        xyzs, rgbs, _ = read_points3D_text(Path(colmap_path) / "points3D.txt")
        storePly(os.path.join(out_dir, scene, 'points3d.ply'), xyzs, rgbs)

        cam_id_to_camera = read_cameras_text(Path(colmap_path) / "cameras.txt")
        im_id_to_image = read_images_text(Path(colmap_path) / "images.txt")

        cameras = {}
        frames = []
        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        depth_size = (256, 192)
        if len(cameras) == 1:
            intrinsics = list(cameras.values())[0]
            fx, fy, cx, cy = intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]

            fovx = 2 * np.arctan(cx / fx)
            trans_info = {'train': {'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": 1000, 'frames': []},
                    'test': {'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": 1000, 'frames': []},
                    'all': { 'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, "depth_scale": 1000, 'frames': []},}
        else:
            print(f"scene {scene} has {len(cameras)} cameras")
            trans_info = {'train': { "depth_scale": 1000, 'frames': []}, 'test': { "depth_scale": 1000, 'frames': []},'all': { "depth_scale": 1000, 'frames': []},}

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())
        select_im_ids = [c for idx, c in enumerate(ordered_im_id) if idx % 5 == 0]


        for idx, im_id in enumerate(select_im_ids):
            if len(cameras) !=1:
                camera_params = cameras[im_id + 1]
                fx, fy, cx, cy = camera_params["fl_x"], camera_params["fl_y"], camera_params["cx"], camera_params["cy"]
                fovx = 2 * np.arctan(cx / fx)
            im_data = im_id_to_image[im_id]
            rotation = qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            if np.isinf(c2w).any():
                print(f'inf pose: {scene, im_id}')
                continue
            dpt_mask_name = im_data.name.replace(".jpg", ".png")
            info = {'file_path': f'input/{im_data.name}', 'transform_matrix': c2w.tolist(), "depth_file_path": f'depth/{dpt_mask_name}',
                    "mask_path": f'input_masks/{im_data.name}'.replace(".jpg", ".png"),'fl_x': fx, 'fl_y': fy, 'cx':cx, 'cy':cy, "camera_angle_x":fovx, 'highres_file_path': f'highres/{im_data.name}'}
            
            if (idx+1) % select_interval_eval ==0:
                trans_info['test']['frames'].append(info)
            else:
                trans_info['train']['frames'].append(info)
            trans_info["all"]['frames'].append(info)

            img_path = os.path.join(rgb_dir, im_data.name)
            rgb = cv2.imread(img_path)
            depth_path = os.path.join(depth_dir, dpt_mask_name)
            rgb = cv2.resize(rgb, depth_size)
            cv2.imwrite(os.path.join(out_rgb_dir, im_data.name), rgb)

            out_depth_path = os.path.join(out_depth_dir, dpt_mask_name)
            shutil.copy2(depth_path, out_depth_path)
            shutil.copy2(os.path.join(mask_dir, dpt_mask_name), os.path.join(out_mask_dir, dpt_mask_name))
            shutil.copy2(os.path.join(rgb_dir, im_data.name), os.path.join(out_highres_dir, im_data.name))


        mesh_path = os.path.join(scene_path, "scans", "mesh_aligned_0.05.ply")
        shutil.copy2(mesh_path, os.path.join(out_dir, scene, scene+'_mesh.ply'))
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        aabb = mesh.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        dimensions = max_bound - min_bound
        scale_sta.append(dimensions[0]*dimensions[1]*dimensions[2])
        
        for split in ['train', 'test', 'all']:
            savedir = os.path.join(out_dir, scene)
            with open(os.path.join(savedir, f'transforms_{split}.json'), 'w', encoding="utf-8") as fp:
                json.dump(trans_info[split], fp, indent=4)

print(np.mean(scale_sta), np.max(scale_sta), np.min(scale_sta))


