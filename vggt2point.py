import roma
import open3d as o3d
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import open3d as o3d
import numpy as np
import json
import os
from PIL import Image
import cv2
from tqdm import tqdm
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import argparse

def convert_dpt2pt(depth, K, pose):
    H, W = depth.shape[0], depth.shape[1]
    x, y = np.meshgrid(np.arange(W, dtype=np.float32),
                            np.arange(H, dtype=np.float32))
    cam_pt = np.stack([(x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], K[2, 2] * np.ones_like(x) * depth, np.ones_like(x)], -1)
    pt = (pose[None, None, :, :]@cam_pt[:, :, :, None]).squeeze()[:, :, :3]
    return pt


parser = argparse.ArgumentParser(description="Save VGGT Point Cloud")
parser.add_argument("--out_path", type=str, default="/media/HDD2/ptmap/scannet/test")
parser.add_argument("--raw_data_path", type=str, default="/media/HDD2/scannet_nerf/scans_test")
args = parser.parse_args()


use_dpt = False  # Set to True to use depth map for point map prediction, False to use point map branch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

scannet_path = args.raw_data_path
out_path = args.out_path
scene_list = os.listdir(scannet_path)
scene_list.sort()  # Sort the scenes to maintain a consistent order

for scene in tqdm(scene_list):
    torch.cuda.empty_cache()
    gt_pts3d = []
    image_names = []
    valid_mask = []
    scene_path = os.path.join(scannet_path, scene)
    if not os.path.isdir(scene_path):
        continue
    print(f"Processing scene: {scene}")
    meta_json = os.path.join(scene_path, "transforms_train.json")
    with open(meta_json, 'r') as f:
        meta_data = json.load(f)
    
    fx, fy, cx, cy = meta_data["fx"], meta_data["fy"], meta_data["cx"], meta_data["cy"]
    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)
    depth_scale = meta_data.get("depth_scale", 1000.0)  # Default depth scale if not provided

    indices = np.linspace(0, len(meta_data["frames"]) - 1, num=50, dtype=int)
    resize_meta_data = [meta_data["frames"][i] for i in indices]  # Select frames based on the indices

    new_world_pose = resize_meta_data[0]["transform_matrix"]
    new_world_pose = np.array(new_world_pose, dtype=np.float32)
    new_world_pose[:3, 1:3] *= -1  # Convert from nerf format to Open3D format
    new_world_transform = np.linalg.inv(new_world_pose)  # Invert the pose to get the camera pose in world coordinates
    # gt_ptslist = []
    for frame in resize_meta_data:
        depth_path = os.path.join(scene_path, frame["depth_file_path"])
        image_path = os.path.join(scene_path, frame["file_path"])
        pose = np.array(frame["transform_matrix"])
        # Convert from nerf format to Open3D format
        pose[:3, 1:3] *= -1
        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float32) / depth_scale  # Convert depth to meters
        mask = depth > 0  # Create a mask for valid depth values

        gt_ptmap = convert_dpt2pt(depth, K, pose)
        gt_ptmap = cv2.resize(gt_ptmap, (518, 392), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (518, 392), interpolation=cv2.INTER_NEAREST).astype(bool)

        gt_pts3d.append(gt_ptmap[None, ...])  # Add batch dimension
        valid_mask.append(mask[None, ...])
        image_names.append(os.path.join(scannet_path, scene, image_path))
# Convert the list of point maps to a single numpy array
    gt_pts3d = np.concatenate(gt_pts3d, axis=0)
    valid_mask = np.concatenate(valid_mask, axis=0)
    gt_pts3d = torch.tensor(gt_pts3d, dtype=torch.float32).to(device)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool).to(device)
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            pred_dict = model(images)
    if use_dpt:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pred_dict["pose_enc"], images.shape[-2:])
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(pred_dict["depth"].squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))
        confs = pred_dict["depth_conf"].squeeze(0)  # Confidence scores for depth predictions
        images = pred_dict["images"].squeeze(0).permute(0, 2, 3, 1)  # Convert to [H, W, C] format
        points = torch.tensor(point_map_by_unprojection, dtype=torch.float32).to(device)  # Convert to tensor and move to device
    else:
        confs = pred_dict["world_points_conf"].squeeze(0)
        images = pred_dict["images"].squeeze(0).permute(0, 2, 3, 1)  # Convert to [H, W, C] format
        points = pred_dict["world_points"].squeeze(0)  # Remove batch dimension

    conf_threshold_reg = torch.quantile(confs, 85 / 100.0)
    conf_mask_reg = confs >= conf_threshold_reg
    final_mask_pred = valid_mask & conf_mask_reg  
    pred_pts_tensor = points[final_mask_pred] # High-confidence predicted points
    gt_pts_tensor_icp = gt_pts3d[final_mask_pred]  # Corresponding GT points
    # Perform rigid registration using Roma
    x = pred_pts_tensor          # High-confidence predicted points (N_pred, 3)
    y = gt_pts_tensor_icp        # Corresponding GT points (N_pred, 3)
    R, t, s = roma.rigid_points_registration(x, y, compute_scaling=True)

    pred_aligned = s * (points.reshape(-1, 3) @ R.T) + t  # Shape: (N_pred, 3)


    colors = images.cpu().numpy().reshape(-1, 3)  # Get colors for the high-confidence points
    pcd_points = pred_aligned.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)  # Downsample the point cloud for better visualization
    # Save the point cloud to a file
    save_path = os.path.join(out_path, scene)
    os.makedirs(save_path, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_path, "point_cloud.ply"), pcd)
