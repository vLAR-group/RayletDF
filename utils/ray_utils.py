import numpy as np
import torch

EPS = 1e-8
def get_rays(H, W, K, use_pixel_centers = True):
    pixel_center = .5 if use_pixel_centers else 0.
    x, y = np.meshgrid(np.arange(W, dtype=np.float32) + pixel_center,
                           np.arange(H, dtype=np.float32) + pixel_center)
    ray_d_cam = np.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(x)], -1)
    rays_d = ray_d_cam / (np.linalg.norm(ray_d_cam, axis=-1, keepdims=True) + EPS)

    return rays_d


def cam_coord2sph(coord, normalize = False, return_radius=False):
    xz_dist = torch.sqrt(coord[..., 2] ** 2 + coord[..., 0] ** 2)
    theta = torch.arctan2(xz_dist, coord[..., 1] + EPS) 
    phi = torch.arctan2(coord[..., 2], (coord[..., 0]) + EPS) 

    if normalize:  # normalize to [-1, 1]
        theta = 2. * (theta / torch.pi) - 1.
        phi = 2. * (phi / torch.pi) - 1.
    if not return_radius:
        return torch.stack([theta, phi], -1)
    else:
        rad = torch.sqrt((coord**2).sum(-1)) 
        return torch.stack([theta, phi, rad], -1)


def cam_sph2coord(sph):
    if sph.shape[-1] == 2:
        r = 1.0
    else:
        r = sph[..., 2]
    theta = sph[..., 0]
    phi = sph[..., 1]
    coord = torch.stack([torch.sin(theta) * torch.cos(phi) * r,
                         torch.cos(theta) * r,
                         torch.sin(theta) * torch.sin(phi) * r], dim=-1)
    return coord  


def gradient(y, x, grad_outputs=None, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=create_graph)[0]
    return grad

def get_surface_normal(t, raydirs):
    dt = gradient(t, raydirs)
    raydirs = raydirs.squeeze()
    dt = dt.squeeze()
    dtdtheta, dtdphi = dt[..., :1], dt[..., 1:]
    sin_theta, cos_theta = torch.sin(raydirs[..., :1]), torch.cos(raydirs[..., :1])
    sin_phi, cos_phi = torch.sin(raydirs[..., 1:]), torch.cos(raydirs[..., 1:])
    dtheta = torch.cat([(dtdtheta * sin_theta + t * cos_theta) * cos_phi,
                        (dtdtheta * cos_theta - t * sin_theta),
                        (dtdtheta * sin_theta + t * cos_theta) * sin_phi,], dim=-1)
    
    dphi = torch.cat([(dtdphi * cos_phi - t * sin_phi) * sin_theta,
                      (dtdphi * cos_theta),
                      (dtdphi * sin_phi + t * cos_phi) * sin_theta], dim=-1)

    normal = torch.cross(dphi, dtheta)
    normal = normal / (torch.linalg.norm(normal+EPS, dim=-1, keepdim=True)+EPS)
    return normal