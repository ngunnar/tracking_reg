import SimpleITK as sitk
import cv2
import numpy as np

from points.inverse_distance import inverse_distance
from tracker.eco_points import FlowPoints
from interpolation.utils import get_nconv_flow, get_tps_flow
from utils.image_Utils import warps, sitk_warps
from utils import result_Utils
from .sikt_methods import bspline_intra_modal_registration, dvf_registration

from interpolation.model import create_model

def store_result(img, segs, flow, result):
    return {'warped': img,
            'warped_seg1': segs[0],
            'warped_seg2': segs[1],
            'warped_seg3': segs[2],
            'flow': flow, 
            'dice_seg1': result[3][0],
            'dice_seg2': result[3][1],
            'dice_seg3': result[3][2],
            'sim': result[0],
            'mse': result[1],
            'diff': result[2]
           }

def run_patient(images, seg_start, seg_end, methods):
    t0 = 0
    print("Reading image {0}...".format(t0))
    image_0 = images[0,...]
    segs_0 = seg_start
    
    frames = []
    frame = {}
    if 'eco' in methods.keys():
        num_of_points = methods['eco']['points']
        print("Initializing {0} trackers...".format(num_of_points))
        p0 = inverse_distance(np.sum(segs_0, axis=0), num_of_points)
        f = FlowPoints(p0, cv2.cvtColor(image_0,cv2.COLOR_GRAY2RGB), True)
        points_0, scores = f._get_points(0, 0)
        frame['eco'] = {
            'points': points_0,
            'scores':scores
        }
    
    frame['index'] = t0
    frame['image'] = image_0
    frame['seg1'] = seg_start[0]
    frame['seg2'] = seg_start[1]
    frame['seg3'] = seg_start[2]
    frame['moving_index'] = t0
    if 'bspline' in methods.keys() or 'demons' in methods.keys():
        image_0_sitk = sitk.GetImageFromArray(image_0.astype('float'))
        segs_0_sitk = [sitk.GetImageFromArray(s.astype(int)) for s in segs_0]
    
    frames.append(frame)
    
    for t in range(t0 + 1, images.shape[0]):
        print("Running image {0}...".format(t))
        image_t = images[t,...]
    
        if 'eco' in methods.keys():
            print("Tracking points ...")
            f.track(cv2.cvtColor(image_t,cv2.COLOR_GRAY2RGB), False)
    
    frame = {}
    frame['index'] = t
    frame['image'] = image_t
    frame['seg1'] = seg_end[0]
    frame['seg2'] = seg_end[1]
    frame['seg3'] = seg_end[2]
    frame['moving_index'] = t0
    
    if 'eco' in methods.keys():
        points_t, scores = f._get_points(t0, t)
        flow_points = points_t - points_0
        frame['eco'] = {'points' :points_t,'scores' : scores}
    
        if 'nconv' in methods['eco'].keys():
            modelpath = methods['eco']['nconv']['model']
            model = create_model('SAME', 256, 256, 'final_model')
            model.load_weights(modelpath)
            eco_nconv_flow = get_nconv_flow(points_t, flow_points, scores, image_t.shape, model)
            image_w, seg_w = warps(eco_nconv_flow, image_0, seg_start)
            result = result_Utils.get_score(image_t, image_w, seg_end, seg_w)            
            frame['eco']['nconv'] = store_result(image_w, seg_w, eco_nconv_flow, result)
        if 'tps' in methods['eco'].keys():
            eco_tps_flow = get_tps_flow(points_t, flow_points, image_t.shape)
            image_w, seg_w = warps(eco_tps_flow, image_0, seg_start)
            result = result_Utils.get_score(image_t, image_w, seg_end, seg_w)
            frame['eco']['tps'] = store_result(image_w, seg_w, eco_tps_flow, result)
            
    if 'bspline' in methods.keys() or 'demons' in methods.keys():
        interp = sitk.sitkBSpline
        image_t_sitk = sitk.GetImageFromArray(image_t.astype('float'))
        seg_end_sitk = [sitk.GetImageFromArray(s.astype(int)) for s in seg_end]
    
    if 'bspline' in methods.keys():
        spacing = methods['bspline']['spacing']
        metric = methods['bspline']['metric']
        transform_ffd_mse = bspline_intra_modal_registration(image_t_sitk, image_0_sitk, spacing, metric)
        image_w, seg_w = sitk_warps(transform_ffd_mse, interp, image_0_sitk, seg_end_sitk)
        result = result_Utils.get_score(image_t, image_w, seg_end, seg_w)
        frame['bspline'] = store_result(image_w, seg_w, None, result)
            
    if 'demons' in methods.keys():
        transform_dvf_demon = dvf_registration(image_t_sitk, image_0_sitk, methods['demons']['intDiffThresholds'], 'demon')
        image_w, seg_w = sitk_warps(transform_dvf_demon, interp, image_0_sitk, seg_end_sitk)
        result = result_Utils.get_score(image_t, image_w, seg_end, seg_w)
        frame['demon'] = store_result(image_w, seg_w, None, result)
    
    frames.append(frame)
    return frames