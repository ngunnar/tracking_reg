
from scipy import ndimage
import numpy as np

def inverse_distance(segmentation_masks, num_of_points, l=1, alpha = 1):
    img_height = segmentation_masks.shape[0]
    dt = ndimage.distance_transform_edt(segmentation_masks == False) + alpha
    dt = (1 + dt/l)**(-alpha)
    probability = (dt) / (np.sum(dt))
    a = np.arange(segmentation_masks.flatten().shape[0])
    points = np.random.choice(a, size=num_of_points, p=probability.flatten(), replace=False)
    xp = np.floor(points/img_height)
    yp = (points/img_height - xp)*img_height
    p0 = np.stack([xp,yp], axis=1).astype('float32')
    return p0