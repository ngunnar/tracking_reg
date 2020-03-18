from .tf_Utils import warp_flow
import numpy as np

def warp(flow_0_t, myo_0, ventricle_0, image_0):
    threshold = 0.5
    myo_t_w = warp_flow(flow=flow_0_t, img=myo_0[None,:,:,None].astype('float'))[0,:,:,0].numpy()
    myo_t_w[myo_t_w>threshold] = 1
    myo_t_w[myo_t_w<=threshold] = 0
    ventricle_t_w = warp_flow(flow=flow_0_t, img=ventricle_0[None,:,:,None].astype('float'))[0,:,:,0].numpy()
    ventricle_t_w[ventricle_t_w>threshold] = 1
    ventricle_t_w[ventricle_t_w<=threshold] = 0
    image_t_w = np.round(warp_flow(flow=flow_0_t, img=image_0[None,:,:,None].astype('float'))[0,...].numpy()).astype(int)
    return myo_t_w, ventricle_t_w, image_t_w[:,:,0]