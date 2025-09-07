
import torch
import os
from tqdm import tqdm
from gaussian_renderer import render_raylet
from utils.general_utils import safe_state, build_invcovariance_from_scaling_rotation
from argparse import ArgumentParser
from arguments import MyModelParams, PipelineParams, get_combined_args
import numpy as np
from utils.loss_utils import l1_loss_mask
from network.RaySurfDNet import RaySurfDNetMLP
from network.spconv_unet import SpUNetBase
from torch.utils.data import DataLoader
from pytorch3d.ops import knn_points
import torch_scatter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from utils.ray_utils import cam_sph2coord, cam_coord2sph
from dataset.train_dataset import SceneGaussianDataset, collate_fn
from utils.system_utils import move_to_device
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def train(dataset, pipe, args):
    TOPK = args.topk
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        print(f"Start running DDP on rank {local_rank}.")
        random.seed(local_rank)
        np.random.seed(local_rank)
        torch.manual_seed(local_rank)
        distributed = True
    else:
        local_rank = 0
        distributed = False
        print("Start running single GPU.")

    save_path = args.model_save_path
    os.makedirs(save_path, exist_ok = True)
    if TENSORBOARD_FOUND and local_rank == 0:
        tb_writer = SummaryWriter(save_path)
    else:
        print("Tensorboard not available: not logging progress")
        tb_writer = None
    first_iter = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    render_net = RaySurfDNetMLP(map_layer_dim=6+39*args.knn_num).to(device)
    pt_net = SpUNetBase(4, 32, channels=(32,64,64,32), layers=(2,2,2,2),).to(device)

    if distributed:
        render_net = DDP(render_net.to(device), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)
        pt_net = DDP(pt_net.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        gs_dataset = SceneGaussianDataset(args.scene_model_dir, args.scene_source_dir, dataset, iterations=args.iterations)
        sampler = torch.utils.data.distributed.DistributedSampler(gs_dataset)
        train_datloader = DataLoader(gs_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=sampler)
        args.dpt_lr*=dist.get_world_size()
        args.iterations = len(train_datloader)
    else:
        gs_dataset = SceneGaussianDataset(args.scene_model_dir, args.scene_source_dir, dataset, iterations=args.iterations)
        train_datloader = DataLoader(gs_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(list(pt_net.parameters()) + list(render_net.parameters()), lr=args.dpt_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.dpt_lr*0.01)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    cur_iteration = first_iter
    ema_loss_for_log = 0.0


    for i, (gaussians, viewpoint_cam, view_rays) in enumerate(train_datloader):
        sph_coord = cam_coord2sph(view_rays.float().to(device), normalize=False).reshape(-1, 1, 2)
        view_rays = view_rays.reshape(-1, 3)
        gaussians, viewpoint_cam, view_rays = move_to_device(gaussians, viewpoint_cam, view_rays, device=device)
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


        NUM_PIXEL = viewpoint_cam.image_height * viewpoint_cam.image_width

        geo_fea = pt_net(pt_inputs)
        cur_iteration += 1
        with torch.no_grad():
            render_pkg = render_raylet(viewpoint_cam, gaussians, pipe, background, topk=TOPK)
            gt_dpt = torch.from_numpy(viewpoint_cam.dist).to(device).flatten()
            mask = gt_dpt > 0.1
            render_idx = render_pkg["depth"].long().permute(1,2,0)
            render_gs = gs_pts[render_idx].reshape(-1, TOPK, 3, 1)
            render_inv_cov = gs_inv_cov[render_idx].reshape(-1, TOPK, 3, 3)
            R_C2W, trans_W2C = torch.from_numpy(viewpoint_cam.R).to(device).float(), torch.from_numpy(viewpoint_cam.T).to(device).float()  

        inds = torch.randperm(NUM_PIXEL)
        iter_step = args.iter_step
        bs_ind = inds[:iter_step]
        batch_raydirs, batch_render_gs, batch_render_inv_cov = sph_coord[bs_ind], render_gs[bs_ind], render_inv_cov[bs_ind]
        ray_dir_cam = cam_sph2coord(batch_raydirs) 
        ray_dir_w = torch.matmul(R_C2W.unsqueeze(0), ray_dir_cam.transpose(1, 2))[:, None]
        cam_org = torch.mv(R_C2W, -trans_W2C)[..., None]
        t_opt = ray_dir_w.transpose(-1, -2)@batch_render_inv_cov@(batch_render_gs - cam_org) / (ray_dir_w.transpose(-1, -2)@batch_render_inv_cov@ray_dir_w)
        hit_world_coord = cam_org + t_opt * ray_dir_w
        ray_dir_world = ray_dir_w.squeeze(-1)

        per_view_loss = []
        optimizer.zero_grad()


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
        dpt_delta = render_pkg_net[...,0]

        if TOPK == 1:
            render_dist = dpt_delta+t_opt.squeeze()
        else:
            unc = render_pkg_net[...,1]
            unc = torch.softmax(unc, dim=-1)
            render_dist = ((dpt_delta+t_opt.squeeze())*unc).sum(-1)
        loss = l1_loss_mask(render_dist, gt_dpt[bs_ind], mask=mask[bs_ind])   
        loss.backward() 
        optimizer.step()
        per_view_loss.append(loss.item())


        with torch.no_grad():
            if local_rank == 0:
                # Progress bar
                ema_loss_for_log = 0.4 * np.mean(per_view_loss) + 0.6 * ema_loss_for_log
                if cur_iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if cur_iteration == args.iterations:
                    progress_bar.close()

        scheduler.step()
        if tb_writer and local_rank == 0:
            tb_writer.add_scalar('train_loss_patches/total_loss', loss, cur_iteration)
        if (cur_iteration in args.save_iterations or (cur_iteration % 20000 == 0)) and (local_rank == 0):
            print("\n[ITER {}] Saving dpt networks".format(cur_iteration))
            if distributed:
                ckpt_dict = {"pt_net":pt_net.module.state_dict(), "render_net": render_net.module.state_dict(), "iteration": cur_iteration,'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            else:
                ckpt_dict = {"pt_net":pt_net.state_dict(), "render_net": render_net.state_dict(), "iteration": cur_iteration,'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(ckpt_dict, "{}/dpt_ckpt_{:07d}.pth".format(save_path, cur_iteration))
    if distributed:
        dist.destroy_process_group()
        print(f"Finished running DDP on rank {local_rank}.")    
    else:
        print("Finished training.")         




if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    model = MyModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--dpt_lr", default=1e-4, type=float)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--testing_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--iter_step", default=24000, type=int, help="number of rays per iteration")
    parser.add_argument("--knn_num", default=5, type=int, help="number of nearest neighbors")
    parser.add_argument("--topk", default=5, type=int, help="number of raylet per ray")
    parser.add_argument("--voxel_size", default=0.05, type=float, help="Voxel size for sparse convolution downsampling")
    parser.add_argument("--model_save_path", required=False, type=str, default="", help="Path to save checkpoints")
    parser.add_argument("--scene_model_dir", nargs="+", required=False, type=str, default=[""], help="Path to the scene point cloud data")
    parser.add_argument("--scene_source_dir", nargs="+", required=False, type=str, default=[""], help="Path to the scene RGBD/pose data")
    args = get_combined_args(parser)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)
    train(model, pipeline, args)

