import numpy as np
import torch

def to_normalmap(x, m=None, white_bkgd=True):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    m = m if m is not None else np.ones_like(x)

    o = 255. * np.ones_like(x) if white_bkgd else np.zeros_like(x)
    xm = x[m[...,0]==1]
    o[m[...,0]==1] = 255 * (xm + 1.) / 2.
    return o.astype(np.uint8)