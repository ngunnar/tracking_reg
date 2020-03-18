
from scipy import ndimage
import numpy as np

def inverse_distance(segmentation_masks, num_of_points):
    img_height = segmentation_masks.shape[0]
    dt = ndimage.distance_transform_edt(segmentation_masks == False) + 1
    probability = (1/dt) / (np.sum(1/dt))
    a = np.arange(segmentation_masks.flatten().shape[0])

    points = np.random.choice(a, size=num_of_points, p=probability.flatten())
    yp = np.floor(points/img_height)
    xp = (points/img_height - yp)*img_height
    p0 = np.stack([xp,yp], axis=1)[:,None,:].astype('float32')    
    return p0