from .tf_Utils import draw_hsv

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

def get_dice_score(seg, gt):
    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def mse(x, y):    
    return mean_squared_error(x, y)

def get_score(image_w, ventricle_w, myo_w,
         image_t, ventricle_t, myo_t):
    ventricle_dice = get_dice_score(ventricle_w, ventricle_t)
    myo_dice = get_dice_score(myo_w, myo_t)
    sim, diff = ssim(image_t, image_w, data_range=np.max(image_w) - np.min(image_w), full=True, multichannel=True)
    mse_error=mse(image_w, image_t)
    return ventricle_dice, myo_dice, sim, mse_error, diff

def plot(image_0, ventricle_0, myo_0,
         image_w, ventricle_w, myo_w,
         image_t, ventricle_t, myo_t,
         flow=None):
    
    ventricle_dice, myo_dice, sim, mse_error, diff = get_score(image_w, ventricle_w, myo_w,
                                                          image_t, ventricle_t, myo_t)
    shape = image_0.shape
    temp_img = np.zeros((shape[0], shape[1]))
    fig, ax = plt.subplots(4,3,figsize=(15,15),sharex=True, sharey=True)        
    ax[0,0].imshow(temp_img, cmap='gray')
    ax[0,0].contour(ventricle_0)    
    ax[0,1].imshow(temp_img, cmap='gray')
    ax[0,1].contour(ventricle_w)    
    ax[0,2].imshow(temp_img, cmap='gray')
    ax[0,2].contour(ventricle_t)    
    print("Ventricle dice {0}".format(ventricle_dice))
    
    ax[1,0].imshow(temp_img, cmap='gray')
    ax[1,0].contour(myo_0)    
    ax[1,1].imshow(temp_img, cmap='gray')
    ax[1,1].contour(myo_w)    
    ax[1,2].imshow(temp_img, cmap='gray')
    ax[1,2].contour(myo_t)    
    print("Myocardium dice {0}".format(myo_dice))
        
    ax[2,0].imshow(image_0, cmap='gray')   
    ax[2,0].contour(myo_0, colors='b')
    ax[2,0].contour(ventricle_0, colors='r')    
    ax[2,1].imshow(image_w, cmap='gray')
    ax[2,1].contour(myo_w, colors='b') 
    ax[2,1].contour(ventricle_w, colors='r')       
    ax[2,2].imshow(image_t, cmap='gray')
    ax[2,2].contour(myo_t, colors='b')    
    ax[2,2].contour(ventricle_t, colors='r')    
    print("SSIM: {0}, MSE: {1}".format(sim, mse_error))
    
    ax[3,0].imshow(image_t, cmap='gray')
    ax[3,0].contour(ventricle_w, colors='r')
    ax[3,0].contour(ventricle_t, colors='b')
    
    ax[3,1].imshow(image_t, cmap='gray')
    ax[3,1].contour(myo_w, colors='r')
    ax[3,1].contour(myo_t, colors='b')
    if flow is not None:       
        ax[3,2].imshow(draw_hsv(flow[None,...])[0,...])
    plt.show()    
    return ventricle_dice, myo_dice, sim, mse_error, diff