import SimpleITK as sitk
import cv2

from points.inverse_distance import inverse_distance
from tracker.eco_points import FlowPoints
from interpolation.utils import get_nconv_flow, get_tps_flow
from utils.image_Utils import warp
from utils import result_Utils
from .sikt_methods import bspline_intra_modal_registration, dvf_registration

def store_result(img, ventricle, myo, flow, result):
    return {'warped': img,
            'warped_ventricle': ventricle,
            'warped_myocardium': myo,
            'flow': flow, 
            'dice_ventricle': result[0],
            'dice_myocardium': result[1],
            'sim': result[2],
            'mse': result[3],
            'diff': result[4]
           }

def run_patient(img_reader, patient, z_pos, methods, start_frame, frames_num):
    t0 = start_frame
    print("Reading image {0}...".format(t0))
    image_0, [ventricle_0, myo_0] = img_reader.read_image(patient, t0, z_pos)
    
    frames = []
    frame = {}
    if 'eco' in methods.keys():
        num_of_points = 3
        print("Initializing {0} trackers...".format(num_of_points))
        p0 = inverse_distance((myo_0 + ventricle_0), num_of_points)
        f = FlowPoints(p0, image_0, True)
        points_0, scores = f._get_points(0, 0)
        frame['eco'] = {
            'points': points_0,
            'scores':scores
        }

    image_0 = cv2.cvtColor(image_0,cv2.COLOR_RGB2GRAY).astype('uint8')
    frame['index'] = t0
    frame['image'] = image_0
    frame['ventricle'] = ventricle_0
    frame['myocardium'] = myo_0
    frame['moving_index'] = t0
    if 'bspline' in methods.keys() or 'demons' in methods.keys():
        image_0_sitk = sitk.GetImageFromArray(image_0.astype('float'))
        ventricle_0_sitk = sitk.GetImageFromArray(ventricle_0.astype(int))
        myo_0_sitk = sitk.GetImageFromArray(myo_0.astype(int))
        frame['image_sitk'] = image_0_sitk
        frame['ventricle_sitk'] = ventricle_0_sitk
        frame['myocardium_sitk'] = myo_0_sitk
    
    frames.append(frame)
    for t in range(start_frame + 1,start_frame + frames_num):
        print("Running image {0}...".format(t))

        image_t, [ventricle_t, myo_t] = img_reader.read_image(patient, t, z_pos)
    
        frame = {}
        frame['index'] = t
        frame['image'] = image_t
        frame['ventricle'] = ventricle_t
        frame['myocardium'] = myo_t
        frame['moving_index'] = t0
    
        if 'eco' in methods.keys():
            print("Tracking points ...")
            f.track(image_t, True)
    
        image_t = cv2.cvtColor(image_t,cv2.COLOR_RGB2GRAY).astype('uint8')
    
        if 'eco' in methods.keys():
            points_t, scores = f._get_points(t0, t)
            flow_points = points_t - points_0
            frame['eco'] = {
                'points' :points_t,
                'scores' : scores            
            }
            if 'nconv' in methods['eco'].keys():
                model = methods['eco']['nconv']['model']
                eco_nconv_flow = get_nconv_flow(points_t, flow_points, scores, image_t.shape, model)
                ventricle_w, myo_w, image_w = warp(eco_nconv_flow, ventricle_0, myo_0, image_0)
                eco_nconv = result_Utils.get_score(image_w, ventricle_w, myo_w, image_t, ventricle_t, myo_t)
                frame['eco']['nconv'] = store_result(image_w, ventricle_w, myo_w, eco_nconv_flow, eco_nconv)
            
            if 'tps' in methods['eco'].keys():
                eco_tps_flow = get_tps_flow(points_t, flow_points, image_t.shape)
                ventricle_w, myo_w, image_w = warp(eco_tps_flow, ventricle_0, myo_0, image_0)
                eco_tps = result_Utils.get_score(image_w, ventricle_w, myo_w, image_t, ventricle_t, myo_t)
                frame['eco']['tps'] = store_result(image_w, ventricle_w, myo_w, eco_tps_flow, eco_tps)
    
        if 'bspline' in methods.keys() or 'demons' in methods.keys():
            image_t_sitk = sitk.GetImageFromArray(image_t.astype('float'))
    
        if 'bspline' in methods.keys():
            spacing = methods['bspline']['spacing']
            metric = methods['bspline']['metric']
            transform_ffd_mse = bspline_intra_modal_registration(image_t_sitk, image_0_sitk, spacing, metric)
            interp = sitk.sitkBSpline
            image_w_sitk = sitk.Resample(image_0_sitk, transform_ffd_mse, interp, 0.0, image_0_sitk.GetPixelID())
            myo_w_sitk = sitk.Resample(myo_0_sitk, transform_ffd_mse, interp, 0.0, image_0_sitk.GetPixelID())
            ventricle_w_sitk = sitk.Resample(ventricle_0_sitk, transform_ffd_mse, interp, 0.0, image_0_sitk.GetPixelID())     
    
            image_w = sitk.GetArrayFromImage(image_w_sitk)
            myo_w = sitk.GetArrayFromImage(myo_w_sitk)
            myo_w[myo_w<0.5] = 0
            myo_w[myo_w>0.5] = 1
            ventricle_w = sitk.GetArrayFromImage(ventricle_w_sitk)
            ventricle_w[ventricle_w<0.5] = 0 
            ventricle_w[ventricle_w>0.5] = 1
            bspline = result_Utils.get_score(image_w, ventricle_w, myo_w, image_t, ventricle_t, myo_t)
    
            frame['bspline'] = store_result(image_w, ventricle_w, myo_w, None, bspline)
        
        if 'demons' in methods.keys():
            transform_dvf_demon = dvf_registration(image_t_sitk, image_0_sitk, methods['demons']['intDiffThresholds'], 'demon')
            interp = sitk.sitkBSpline
            image_w_sitk = sitk.Resample(image_0_sitk, transform_dvf_demon, interp, 0.0, image_0_sitk.GetPixelID())
            myo_w_sitk = sitk.Resample(myo_0_sitk, transform_dvf_demon, interp, 0.0, image_0_sitk.GetPixelID())
            ventricle_w_sitk = sitk.Resample(ventricle_0_sitk, transform_dvf_demon, interp, 0.0, image_0_sitk.GetPixelID())     
            
            image_w = sitk.GetArrayFromImage(image_w_sitk)
            myo_w = sitk.GetArrayFromImage(myo_w_sitk)
            myo_w[myo_w<0.5] = 0
            myo_w[myo_w>0.5] = 1
            ventricle_w = sitk.GetArrayFromImage(ventricle_w_sitk)
            ventricle_w[ventricle_w<0.5] = 0 
            ventricle_w[ventricle_w>0.5] = 1
        
            demon = result_Utils.get_score(image_w, ventricle_w, myo_w, image_t, ventricle_t, myo_t)
        
            frame['demon'] = store_result(image_w, ventricle_w, myo_w, None, demon)
        frames.append(frame)
    return frames