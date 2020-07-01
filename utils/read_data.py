import math
import matplotlib.path
import scipy.io
import numpy as np
import cv2 as cv
import sys
from scipy import interpolate
import glob
import re

def grp(file):
    r = re.findall('\d+.mat',file)
    assert len(r) == 1
    r = r[0].split('.')[0]    
    return int(r)

class ImageReader():

    def __init__(self, data_path, seg_path, img_heigth, img_width):
        self.data_files = glob.glob(data_path + '/*.mat')
        self.data_files.sort(key=lambda l: grp(l))
        self.seg_files = glob.glob(seg_path + '/*.mat')
        self.seg_files.sort(key=lambda l: grp(l))
        self.img_heigth = img_heigth
        self.img_width = img_width    

    def read_image(self, patient, frame = 0, z_pos = 4):
        image_file = self.data_files[patient]
        seg_file = self.seg_files[patient]
        gray = scipy.io.loadmat(image_file)['sol_yxzt'][:,:,z_pos, frame][...,None]
        seg_data = scipy.io.loadmat(seg_file)['manual_seg_32points'][z_pos, frame]  
    
        if seg_data[0,0] == -9999:
            return image
       
        sx, sy = seg_data.shape    
        half = (sx)//2  
        endo_data = seg_data[:half,:]
        epi_data = seg_data[half+1:,:]

        image = cv.cvtColor(gray,cv.COLOR_GRAY2RGB).astype('uint8') 
    
        if image.shape != (self.img_heigth,self.img_width, 3):
            org_shape = image.shape
            image = cv.resize(image, (self.img_heigth,self.img_width))
        
            endo_data[:,0] = endo_data[:,0] * self.img_heigth/org_shape[0]
            endo_data[:,1] = endo_data[:,1] * self.img_width/org_shape[1]
            epi_data[:,0] = epi_data[:,0] * self.img_heigth/org_shape[0]
            epi_data[:,1] = epi_data[:,1] * self.img_width/org_shape[1]
    
        left = np.asarray([0,0])
        right = np.asarray([self.img_heigth,self.img_width])
        x = np.arange(math.ceil(left[0]), math.floor(right[0]))
        y = np.arange(math.ceil(left[1]), math.floor(right[1]))
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))

        endo_path = matplotlib.path.Path(endo_data)
        ventricle_mask = endo_path.contains_points(points)
        ventricle_mask.shape = xv.shape        
        
        epi_path = matplotlib.path.Path(epi_data)
        myo_mask = epi_path.contains_points(points)
        myo_mask.shape = xv.shape
        myo_mask = (myo_mask.astype(int) - ventricle_mask.astype(int)).astype('bool')
        
        endo = np.zeros((self.img_heigth, self.img_width))
        epi = np.zeros((self.img_heigth, self.img_width))
        for p in endo_data:
            endo[int(p[1]), int(p[0])] = 1
            
        for p in epi_data:
            epi[int(p[1]), int(p[0])] = 1

        return image, [ventricle_mask, myo_mask], [endo, epi]