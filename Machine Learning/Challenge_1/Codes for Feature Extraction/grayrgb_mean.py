# This code is used to extract graylevel & RGB mean values from each images. 

import cv2 as cv
import cv2
import numpy as np

def gray_rgb_mean(img):
    
    grgb_features = []
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    (img_B, img_G, img_R) = cv.split(img)
    
    gray_mean = np.mean(img_gray)
    blue_mean = np.mean(img_B)
    green_mean = np.mean(img_G)
    red_mean = np.mean(img_R)
    
    grgb_features.extend([gray_mean, blue_mean, green_mean, red_mean])
    
    return grgb_features
