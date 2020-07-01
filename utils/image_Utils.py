from .tf_Utils import warp_flow
import numpy as np
import SimpleITK as sitk

def warps(flow_0_t, image_0, segs_0):
    threshold = 0.5
    segs_w = []
    for s in segs_0:
        s_w = warp_flow(flow=flow_0_t, img=s[None,:,:,None].astype('float'))[0,:,:,0].numpy()
        s_w[s_w>threshold] = 1
        s_w[s_w<=threshold] = 0
        segs_w.append(s_w)
        
    image_w = np.round(warp_flow(flow=flow_0_t, 
                                 img=image_0[None,:,:,None].astype('float')
                                )[0,:,:,0].numpy()).astype(int)
    return image_w, segs_w

def sitk_warps(reg, interp, image_0, segs_0):
    threshold = 0.5
    image_w = sitk.Resample(image_0, reg, interp, 0.0, image_0.GetPixelID())
    image_w = sitk.GetArrayFromImage(image_w)
    segs_w = []
    for s in segs_0:
        s_w = sitk.Resample(s, reg, interp, 0.0, image_0.GetPixelID())
        s_w = sitk.GetArrayFromImage(s_w)
        s_w[s_w>threshold] = 1
        s_w[s_w<=threshold] = 0
        segs_w.append(s_w)
    return image_w, segs_w