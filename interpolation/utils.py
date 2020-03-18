import numpy as np
import scipy

def get_nconv_flow(points_t, flow_points, scores, shape, model):
    c = np.zeros((shape[0], shape[1]))
    x1 = np.zeros((shape[0], shape[1]))
    x2 = np.zeros((shape[0], shape[1]))
    for idx in range(len(points_t)):
        if points_t[idx,0] >= shape[0] or points_t[idx,1] >= shape[1] or points_t[idx,0] < 0 or points_t[idx,1] < 0:
            continue
        c[points_t[idx,1], points_t[idx,0]] = 1 #np.mean(scores[idx,:])
        x2[points_t[idx,1], points_t[idx,0]] = flow_points[idx,0]
        x1[points_t[idx,1], points_t[idx,0]] = flow_points[idx,1]
        
    x = np.concatenate((x1[...,None],x2[...,None]),axis=-1)
    out = model.predict([x[None, ...], c[None,:,:,None]])
    dense_flow = out[0,:,:,0:2]    
    return dense_flow

def get_tps_flow(points_t, flow_points, shape):    
    b = []
    fp = points_t.copy().tolist()
    fpoints = []
    flowpoints = []
    for i in range(len(fp)):
        p = fp[i]
        if p not in b:
            fpoints.append(p)
            flowpoints.append(flow_points[i])   
        b.append(p)       
    
    fpoints = np.asarray(fpoints)
    flowpoints = np.asarray(flowpoints)
    x = fpoints[:,0]
    y = fpoints[:,1]        
        
    z_x = flowpoints[:,0]
    z_y = flowpoints[:,1]

    interp_x = scipy.interpolate.Rbf(x, y, z_x, function='thin_plate')
    interp_y = scipy.interpolate.Rbf(x, y, z_y, function='thin_plate')

    yi, xi = np.mgrid[0:shape[0]:1, 0:shape[1]:1]
    z_xi = interp_x(xi, yi)
    z_yi = interp_y(xi, yi)

    flow = np.stack([z_yi, z_xi], axis=-1)[None,...]
    return flow