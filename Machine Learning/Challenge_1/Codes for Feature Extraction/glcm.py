# This code is used to extract 5 GLCM features from each images. 

import cv2
import cv2 as cv
from skimage.feature import greycomatrix, greycoprops
import numpy as np

def glcmFeatures(img):
    glcm_features = []
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    glcm = greycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    GLCM_energy = greycoprops(glcm, 'energy')[0, 0]
    GLCM_correlation = greycoprops(glcm, 'correlation')[0, 0]
    GLCM_homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    GLCM_contrast = greycoprops(glcm, 'contrast')[0, 0]
    GLCM_dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    
    glcm_features.extend([GLCM_energy, GLCM_correlation, GLCM_homogeneity, GLCM_contrast, GLCM_dissimilarity])
    
    return glcm_features
