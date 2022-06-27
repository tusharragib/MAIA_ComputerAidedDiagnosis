# This code is used to extract 8 gabor features from each images. 

import numpy as np
import cv2 as cv

def gabor_features(img):
    ksize = 10
    sigma = 3
    gamma = 0.5
    phi = 0
    
    i = 0    
    gabor_features =[]
    
    for lambd in (1*np.pi/8, 1*np.pi/4):
        for theta in (0 , 1*np.pi/4, 1*np.pi/2, 3*np.pi/4):
            kernel = cv.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, phi, ktype=cv.CV_32F)
            fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
            i = i+1
            gabor = fimg.mean()
            
            gabor_features.extend([gabor])

    return gabor_features
