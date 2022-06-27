# This code is used to extract 27 lbp features from each images. 

import cv2
import cv2 as cv
from skimage.feature import local_binary_pattern
import numpy as np

def lbp_features(img):

    eps = 1e-7

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    lbp = local_binary_pattern(img_gray, 24, 3, method="uniform")
    
    (hist, _) = np.histogram(lbp.ravel(), bins = 27, range=(0, 27))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps) #eps is a very small value in case if hist contains zero value.
    
    hist = hist.tolist()
    
    return hist
