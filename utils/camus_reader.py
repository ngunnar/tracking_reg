import SimpleITK as sitk
import cv2
import numpy as np

def read_data(filename, shape=None):
    itkimage = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(itkimage)
    if shape != None:
        return np.asarray([cv2.resize(img[t,...], shape) for t in range(img.shape[0])])
    else:
        return img

def read_seg(filename, no_seg, shape=None):
    seg = read_data(filename)
    seg_list = np.asarray([(seg[0,...] == i).astype('uint8') for i in range(1, no_seg+1)])
    if shape != None:
        return np.asarray([cv2.resize(s, shape) for s in seg_list])
    else:
        return seg_list

def camus_reader(f_image, f_contour_start, f_contour_end, no_seg, shape=None):
    images = read_data(f_image, shape)
    contour_start = read_seg(f_contour_start, no_seg, shape)
    contour_end = read_seg(f_contour_end, no_seg, shape)
    return images, contour_start, contour_end