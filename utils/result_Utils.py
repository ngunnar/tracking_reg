import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
from skimage import feature

from skimage import feature
def getDistancesFromAtoB(a, b):
    # Quick nearest-neighbor lookup
    kdTree = KDTree(a, leafsize=100)
    return kdTree.query(b, k=1, eps=0, p=2)[0]

def hausdorff_95(gt, image_w):
    edges = feature.canny(gt, sigma=3)
    edges_w = feature.canny(image_w.astype('uint8'), sigma=3)
    
    w_coordinates   = [x.tolist() for x in np.transpose( np.flipud( np.nonzero(edges_w) ))]
    gt_coordinates = [x.tolist() for x in np.transpose( np.flipud( np.nonzero(edges) ))]
    d_wTogt = getDistancesFromAtoB(w_coordinates, gt_coordinates)
    d_gtTow = getDistancesFromAtoB(gt_coordinates, w_coordinates) 
    return max(np.percentile(d_wTogt, 95), np.percentile(d_gtTow, 95))

def get_dice_score(seg, gt):
    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def jacobian(flow, plus = False):
    du = np.gradient(flow[...,0])
    dv = np.gradient(flow[...,1])

    detJ = np.zeros((flow.shape[0],flow.shape[1]))
    for x in range(flow.shape[0]):
        for y in range(flow.shape[1]):
            J11 = 1 + du[0][x,y] if plus else 1 - du[0][x,y]
            J12 = du[1][x,y]
            J21 = dv[0][x,y]
            J22 = 1 + dv[1][x,y] if plus else 1 - dv[1][x,y]
            J = np.array([[J11, J12],[J21,J22]])
            detJ[x,y] = np.linalg.det(J)
    return np.std(detJ), detJ

def mse(x, y):    
    return mean_squared_error(x, y)

def get_score(image_gt, image_w, segs_gt, segs_w):
    segs_dice = [get_dice_score(segs_w[i], segs_gt[i]) for i in range(len(segs_gt))]    
    sim, diff = ssim(image_gt, image_w, data_range=np.max(image_w) - np.min(image_w), full=True, multichannel=True)
    mse_error=mse(image_w, image_gt)
    h95 = hausdorff_95(image_gt, image_w)
    return sim, mse_error, diff, segs_dice, h95