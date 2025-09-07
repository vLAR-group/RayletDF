import numpy as np

EPS = 1e-6
def complete_ray_metric(pred, gt, mask=None):
    if mask is None:
        mask = gt > 0.1

    thresh = np.maximum((gt[mask] / (pred[mask] + EPS)), (pred[mask] / gt[mask]))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    ade = np.sum(np.abs(pred - gt) * mask) / np.sum(mask) 
    rmse = np.sqrt(np.sum(np.square(pred - gt) * mask) / np.sum(mask))
    rel_ade = (np.abs(pred - gt)[mask] / gt[mask])
    rel_ade = np.mean(rel_ade)
    sq_rel_ade = (np.square(pred - gt)[mask] / gt[mask])
    sq_rel_ade = np.mean(sq_rel_ade)
    return ade, rmse, rel_ade, sq_rel_ade, a1, a2, a3
