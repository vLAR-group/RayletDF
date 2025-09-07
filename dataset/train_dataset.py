import os
import json
import torch
from torch.utils.data import Dataset
from random import randint
import numpy as np
from plyfile import PlyData
from PIL import Image
from pathlib import Path
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov, fov2focal
from utils.graphics_utils import convert_d, BasicPointCloud
from utils.camera_utils import loadCam, CameraInfoDpt
from utils.ray_utils import get_rays
from utils.sh_utils import RGB2SH
from scene import GaussianModel

class SceneGaussianDataset(Dataset):
    def __init__(self, scene_model_dir, scene_source_dir, model_args, iterations = 300000):
        self.scene_model_dir = []
        self.scene_source_dir = []
        for model_dir, source_dir in zip(scene_model_dir, scene_source_dir):
            scenes_list = os.listdir(source_dir)
            scenes_list.sort()
            for scene in scenes_list:
                single_scene_model_dir = os.path.join(model_dir, scene)
                single_scene_source_dir = os.path.join(source_dir, scene)
                assert os.path.exists(single_scene_model_dir)
                assert os.path.exists(single_scene_source_dir)
                self.scene_model_dir.append(single_scene_model_dir)
                self.scene_source_dir.append(single_scene_source_dir)

        self.gaussian_args = model_args
        self.device = model_args.data_device 
        self.transformsfile = "transforms_train.json"
        self.iterations = iterations
        self.length = len(self.scene_model_dir)
        print("Training scenes", self.length)

    def __len__(self):
        return self.iterations
    
    def load_gaussian(self, path, gaussians):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(gaussians.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians._xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        gaussians._features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        gaussians._features_rest = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        gaussians._opacity = torch.tensor(opacities, dtype=torch.float, device=self.device)
        gaussians._scaling = torch.tensor(scales, dtype=torch.float, device=self.device)
        gaussians._rotation = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        gaussians.active_sh_degree = gaussians.max_sh_degree
        return gaussians

    def load_viewpoint(self, path):
        with open(os.path.join(path, self.transformsfile)) as json_file:
            contents = json.load(json_file)
            dpt_scale = contents["depth_scale"]
            frames = contents["frames"]
        frames_len = len(frames)
        viewpoint_idx = randint(0, frames_len-1)
        frame = frames[viewpoint_idx]
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]
        else:
            fovx = frame["camera_angle_x"]
        cam_name = os.path.join(path, frame["file_path"])
        dpt_name = os.path.join(path, frame["depth_file_path"])

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        if c2w.shape[0] != 4:
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if self.gaussian_args.white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        depth_path = os.path.join(path, dpt_name)
        depth = np.array(Image.open(depth_path)) / dpt_scale
        ray = convert_d(depth, fov2focal(fovx, image.size[0]), out='dist')

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx
        return CameraInfoDpt(uid=viewpoint_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ray = ray,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])

    def __getitem__(self, rand_idx):
        idx = randint(0, self.length-1)
        gaussians = GaussianModel(self.gaussian_args.sh_degree)
        model_path = self.scene_model_dir[idx]
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        gaussians = self.load_gaussian(os.path.join(model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"), gaussians)
        source_path = self.scene_source_dir[idx]
        cam_info = self.load_viewpoint(source_path)
        viewpoint = loadCam(self.gaussian_args, cam_info.uid, cam_info, resolution_scale=1.0)

        H, W, focal = viewpoint.image_height, viewpoint.image_width, viewpoint.focal
        K = np.array([[focal, 0, W//2], [0, focal, H//2], [0, 0, 1]])
        view_rays = get_rays(H, W, K)
        view_rays = torch.tensor(view_rays)
        return gaussians, viewpoint, view_rays
    
def collate_fn(batch):
    return batch[0]


class ScenePointCloudDataset(Dataset):
    def __init__(self, scene_model_dir, scene_source_dir, model_args, iterations = 300000):
        self.scene_model_dir = []
        self.scene_source_dir = []
        for model_dir, source_dir in zip(scene_model_dir, scene_source_dir):
            scenes_list = os.listdir(source_dir)
            scenes_list.sort()
            for scene in scenes_list:
                single_scene_model_dir = os.path.join(model_dir, scene)
                single_scene_source_dir = os.path.join(source_dir, scene)
                assert os.path.exists(single_scene_model_dir)
                assert os.path.exists(single_scene_source_dir)
                self.scene_model_dir.append(single_scene_model_dir)
                self.scene_source_dir.append(single_scene_source_dir)

        self.gaussian_args = model_args
        self.device = model_args.data_device 
        self.transformsfile = "transforms_train.json"
        self.iterations = iterations
        self.length = len(self.scene_model_dir)
        print("Training scenes", self.length)

    def __len__(self):
        return self.iterations
    
    def create_from_pcd(self, pcd, gaussians):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = torch.zeros((fused_color.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1
        gaussians._xyz = fused_point_cloud
        gaussians._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        gaussians._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        gaussians._rotation = rots
        gaussians.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), device=self.device)
        return gaussians

    def load_viewpoint(self, path):
        with open(os.path.join(path, self.transformsfile)) as json_file:
            contents = json.load(json_file)
            dpt_scale = contents["depth_scale"]
            frames = contents["frames"]
        frames_len = len(frames)
        viewpoint_idx = randint(0, frames_len-1)
        frame = frames[viewpoint_idx]
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]
        else:
            fovx = frame["camera_angle_x"]
        cam_name = os.path.join(path, frame["file_path"])
        dpt_name = os.path.join(path, frame["depth_file_path"])

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        if c2w.shape[0] != 4:
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if self.gaussian_args.white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        depth_path = os.path.join(path, dpt_name)
        depth = np.array(Image.open(depth_path)) / dpt_scale
        ray = convert_d(depth, fov2focal(fovx, image.size[0]), out='dist')

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx
        return CameraInfoDpt(uid=viewpoint_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ray = ray,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])
    

    def fetchPly(self, path):
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        if "red" in vertices:
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        else:
            colors = np.ones_like(positions)
        if "nx" in vertices:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        else:
            normals = np.zeros_like(positions)
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    def __getitem__(self, rand_idx):
        idx = randint(0, self.length-1)
        gaussians = GaussianModel(self.gaussian_args.sh_degree)
        source_path = self.scene_source_dir[idx]
        model_path = self.scene_model_dir[idx]
        sparse_pts = os.path.join(model_path,"point_cloud.ply")
        pcd = self.fetchPly(sparse_pts)
        gaussians = self.create_from_pcd(pcd, gaussians)
        cam_info = self.load_viewpoint(source_path)
        viewpoint = loadCam(self.gaussian_args, cam_info.uid, cam_info, resolution_scale=1.0)

        H, W, focal = viewpoint.image_height, viewpoint.image_width, viewpoint.focal
        K = np.array([[focal, 0, W//2], [0, focal, H//2], [0, 0, 1]])
        view_rays = get_rays(H, W, K)
        view_rays = torch.tensor(view_rays)
        return gaussians, viewpoint, view_rays