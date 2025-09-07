
import torch
import os
from tqdm import tqdm
from gaussian_renderer import render_raylet
from utils.general_utils import safe_state, build_invcovariance_from_scaling_rotation
from argparse import ArgumentParser
from arguments import MyModelParams, PipelineParams, get_combined_args
import numpy as np
from network.RaySurfDNet import RaySurfDNetMLP
from network.spconv_unet import SpUNetBase
from torch.utils.data import DataLoader
from pytorch3d.ops import knn_points
import torch_scatter
from utils.ray_utils import cam_sph2coord, cam_coord2sph, get_surface_normal
from dataset.test_dataset import SceneGaussianDataset, collate_fn
from utils.vis_utils import to_normalmap
from utils.metrics_utils import complete_ray_metric
import math
import json
import imageio

def test(dataset, pipe, args):
    TOPK = args.topk
    ckpt = torch.load(args.dpt_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    render_net = RaySurfDNetMLP(map_layer_dim=6+39*args.knn_num).to(device)
    pt_net = SpUNetBase(4, 32, channels=(32,64,64,32), layers=(2,2,2,2),).to(device)
    render_net.load_state_dict(ckpt['render_net'])
    pt_net.load_state_dict(ckpt['pt_net'])
    render_net.eval()
    pt_net.eval()

    gs_dataset = SceneGaussianDataset(args.scene_model_dir, args.scene_source_dir, dataset, iterations=args.iterations, split=args.split)
    test_datloader = DataLoader(gs_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    progress_bar = tqdm(range(0, len(test_datloader)), desc="Testing progress")
    for i, (gaussians, viewpoint_cam_list, view_rays, model_path) in enumerate(test_datloader):
        sph_coord = cam_coord2sph(view_rays.float().to(device), normalize=False).reshape(-1, 1, 2)
        view_rays = view_rays.reshape(-1, 3)

        view_rays = view_rays.to(device)
        for key in gaussians.__dict__:
            if key.startswith("_"):
                setattr(gaussians, key, getattr(gaussians, key).to(device))  

        gs_pts = gaussians.get_xyz
        gs_rot_qua = gaussians._rotation
        gs_scale = gaussians.get_scaling
        gs_ops = gaussians.get_opacity
        gs_inv_cov = build_invcovariance_from_scaling_rotation(gs_scale, scaling_modifier = 1, rotation=gs_rot_qua)


        aabb_min_coords = torch.min(gs_pts, dim=0).values
        aabb_max_coords = torch.max(gs_pts, dim=0).values
        centers = (aabb_max_coords + aabb_min_coords) / 2
        norm_gs_pts = (gs_pts - centers)

        pt_inputs = {}
        pt_inputs['features'] = torch.cat((norm_gs_pts, gs_ops), dim=-1)

        sampleDl = args.voxel_size
        originCorner = torch.floor(norm_gs_pts.min(0)[0] * (1/sampleDl)) * sampleDl
        coords = torch.div(norm_gs_pts - originCorner, sampleDl, rounding_mode="trunc").int()
        unq, unq_inv = torch.unique(coords, return_inverse=True, dim=0)
        unq = unq.type(torch.int64)
        ds_feats = torch_scatter.scatter_mean(pt_inputs["features"], unq_inv, dim=0)
        ds_pts = torch_scatter.scatter_mean(norm_gs_pts, unq_inv, dim=0)
        pt_inputs['grid_coord'] = unq
        pt_inputs["bs_id"] = torch.zeros_like(unq[:, 0])
        pt_inputs["features"] = ds_feats
        pt_inputs["ds_pts"] = ds_pts
        geo_fea = pt_net(pt_inputs)

        ades, rmses, rel_ades, sq_rel_ades = [], [], [], []  
        a1s, a2s, a3s = [], [], []
        image_names = []
        full_dict = {args.method_name:{}}
        per_view_dict = {args.method_name:{}}

        render_dist_path = os.path.join(model_path, args.split, args.method_name, "render_dist_net")
        dataset_name, scene_name = model_path.split("/")[-3], model_path.split("/")[-1]
        scene_dir = os.path.join(os.path.dirname(args.dpt_model_path), dataset_name, args.split, scene_name)
        render_normal_path = os.path.join(model_path, args.split, args.method_name, "render_normal_net")

        os.makedirs(scene_dir, exist_ok = True)
        os.makedirs(render_dist_path, exist_ok = True)
        os.makedirs(render_normal_path, exist_ok = True)
        for idx, viewpoint_cam in enumerate(viewpoint_cam_list):
            image_names.append(viewpoint_cam.image_name)
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to(device)
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.to(device)
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.to(device)
            with torch.no_grad():
                render_pkg = render_raylet(viewpoint_cam, gaussians, pipe, background, topk=TOPK)
                render_idx = render_pkg["depth"].long().permute(1,2,0)
                render_gs = gs_pts[render_idx].reshape(-1, TOPK, 3, 1)
                render_inv_cov = gs_inv_cov[render_idx].reshape(-1, TOPK, 3, 3)
                R_C2W, trans_W2C = torch.from_numpy(viewpoint_cam.R).to(device).float(), torch.from_numpy(viewpoint_cam.T).to(device).float()  
            NUM_PIXEL = viewpoint_cam.image_height * viewpoint_cam.image_width
            out_dist, out_normal = [], []
            iter_step = 400000 if not args.normal else 100000
            iter_num = math.ceil(NUM_PIXEL / iter_step)
            for i in range(0, iter_num):
                batch_raydirs, batch_render_gs, batch_render_inv_cov = sph_coord[i*iter_step:(i+1)*iter_step], \
                                render_gs[i*iter_step:(i+1)*iter_step], render_inv_cov[i*iter_step:(i+1)*iter_step]
                if args.normal:
                    batch_raydirs.requires_grad_(True)

                ray_dir_cam = cam_sph2coord(batch_raydirs) 
                ray_dir_w = torch.matmul(R_C2W.unsqueeze(0), ray_dir_cam.transpose(1, 2))[:, None]
                cam_org = torch.mv(R_C2W, -trans_W2C)[..., None]
                t_opt = ray_dir_w.transpose(-1, -2)@batch_render_inv_cov@(batch_render_gs - cam_org) / (ray_dir_w.transpose(-1, -2)@batch_render_inv_cov@ray_dir_w)
                hit_world_coord = cam_org + t_opt * ray_dir_w
                ray_dir_world = ray_dir_w.squeeze(-1)

                hit_world_coord = (hit_world_coord.squeeze(-1) - centers)
                query = hit_world_coord.reshape(-1, 3)
                knn_res = knn_points(query.unsqueeze(0), ds_pts.unsqueeze(0), K=args.knn_num)
                knn_dists, knn_inds = torch.sqrt(knn_res.dists.squeeze()).reshape(-1, TOPK, args.knn_num), knn_res.idx.squeeze().reshape(-1, TOPK, args.knn_num)
                knn_dists = (knn_dists + 1e-8).unsqueeze(-1)
                knn_feats = geo_fea[knn_inds].flatten(-2)
                knn_pts = ds_pts[knn_inds].flatten(-2)
                knn_vec = (pt_inputs["ds_pts"][knn_inds] - hit_world_coord.unsqueeze(-2))
                knn_dir = (knn_vec / knn_dists)
                knn_rel = torch.cat([knn_dir, knn_dists], dim=-1).flatten(-2)

                render_pkg_net = render_net(ray_dir_world, hit_world_coord, knn_pts, knn_rel, knn_feats)
                dist_delta = render_pkg_net[...,0]

                if TOPK == 1:
                    render_dist = dist_delta+t_opt.squeeze()
                else:
                    unc = render_pkg_net[...,1]
                    unc = torch.softmax(unc, dim=-1)
                    render_dist = ((dist_delta+t_opt.squeeze())*unc).sum(-1)

                if args.normal:
                    normal = get_surface_normal(render_dist[...,None], batch_raydirs).detach()
                    out_normal.append(normal)
                render_dist = render_dist.detach()
                out_dist.append(render_dist)
            
            h, w = viewpoint_cam.dist.shape  
            out_dist = torch.cat(out_dist).reshape(h, w)
            out_dist = out_dist.squeeze().cpu().numpy()
            if args.normal:
                out_normal = torch.cat(out_normal).reshape(h, w, 3)
                out_normal = -out_normal
                render_normal = to_normalmap(out_normal).squeeze()
                imageio.imwrite(os.path.join(render_normal_path, viewpoint_cam.image_name + ".png"), render_normal)
            if args.save_dist:
                np.save(os.path.join(render_dist_path, '{0:05d}'.format(idx) + ".npy"), out_dist.astype(np.float16))

            ade, rmse, rel_ade, sq_rel_ade, a1, a2, a3 = complete_ray_metric(out_dist, viewpoint_cam.dist)
            ades.append(ade) 
            rmses.append(rmse)
            rel_ades.append(rel_ade)
            sq_rel_ades.append(sq_rel_ade)
            a1s.append(a1)
            a2s.append(a2)
            a3s.append(a3)

        progress_bar.set_postfix({"Scene":f"{scene_name}","L1": f"{np.mean(ades):.{5}f}"})
        progress_bar.update(1)
        full_dict[args.method_name].update({ "ADE": torch.tensor(ades).mean().item(),
                                        "RMSE": torch.tensor(rmses).mean().item(),
                                        "REL_ADE": torch.tensor(rel_ades).mean().item(),
                                        "SQ_REL_ADE": torch.tensor(sq_rel_ades).mean().item(),
                                        "A1": torch.tensor(a1s).mean().item(),
                                        "A2": torch.tensor(a2s).mean().item(),
                                        "A3": torch.tensor(a3s).mean().item(),
                                        "ADE_std": torch.tensor(ades).std().item(),
                                        "RMSE_std": torch.tensor(rmses).std().item(),
                                        "REL_ADE_std": torch.tensor(rel_ades).std().item(),
                                        "SQ_REL_ADE_std": torch.tensor(sq_rel_ades).std().item(),
                                        "A1_std": torch.tensor(a1s).std().item(),
                                        "A2_std": torch.tensor(a2s).std().item(),
                                        "A3_std": torch.tensor(a3s).std().item(),}
                                                )
        
        per_view_dict[args.method_name].update({
                                                "ADE": {name: ade for ade, name in zip(torch.tensor(ades).tolist(), image_names)},
                                                "RMSE": {name: rmse for rmse, name in zip(torch.tensor(rmses).tolist(), image_names)},
                                                "REL_ADE": {name: rel_ade for rel_ade, name in zip(torch.tensor(rel_ades).tolist(), image_names)},
                                                "SQ_REL_ADE": {name: sq_rel_ade for sq_rel_ade, name in zip(torch.tensor(sq_rel_ades).tolist(), image_names)},
                                                "A1": {name: a1 for a1, name in zip(torch.tensor(a1s).tolist(), image_names)},
                                                "A2": {name: a2 for a2, name in zip(torch.tensor(a2s).tolist(), image_names)},
                                                "A3": {name: a3 for a3, name in zip(torch.tensor(a3s).tolist(), image_names)},
                                                    }
                                            )
    
        
        with open(scene_dir + "/"+args.split+"_results_net.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(scene_dir + "/"+args.split+"_per_view_net.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    progress_bar.close()



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = MyModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dpt_model_path", required=True, type=str, default="")
    parser.add_argument("--method_name", type=str, default="ours_30000")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--normal", type=bool, default=False, help="Whether to derive normal maps from RayletDF")
    parser.add_argument("--save_dist", type=bool, default=False, help="Whether to save the distance maps")
    parser.add_argument("--knn_num", default=5, type=int)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--voxel_size", default=0.05, type=float)
    parser.add_argument("--scene_model_dir", nargs="+", required=False, type=str, default=[""])
    parser.add_argument("--scene_source_dir", nargs="+", required=False, type=str, default=[""])
    args = get_combined_args(parser)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)
    if args.dpt_model_path is not "":
        if args.normal:
            test(model, pipeline, args)
        else:
            with torch.no_grad():
                test(model, pipeline, args)
    else:
        print("Please specify the path of the dpt model")
        exit()

