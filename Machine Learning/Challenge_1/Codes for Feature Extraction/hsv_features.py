# This code is used to extract 9 hsv features from each images. 

import cv2 as cv
import cv2
import numpy as np

def hsv_color_moments(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  #[:,:,0]
    h, s, v = cv2.split(img_hsv)
    color_feature = []
    
    #One-stage moment (mean mean)
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])  #  One-stage moment placement feature array
    
    #Secondary moments (standard difference STD)
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])  #  Second order moment placed in a feature array
    
    #Three-order moment (slope Skewness)
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])  #  Three-order moments in the feature array
    
    return color_feature
