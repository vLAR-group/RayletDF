#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def move_to_device(gaussians, viewpoint_cam, view_rays, device):
    view_rays = view_rays.to(device)
    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to(device)
    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.to(device)
    viewpoint_cam.camera_center = viewpoint_cam.camera_center.to(device)
    for key in gaussians.__dict__:
        if key.startswith("_"):
            setattr(gaussians, key, getattr(gaussians, key).to(device))  

    return gaussians, viewpoint_cam, view_rays