# This code is used to extract 7 hu moment features from each images. 

import cv2
import cv2 as cv

def hu_moments(img):
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    hu_0 = cv.HuMoments(cv.moments(img_gray)).flatten()
    hu = hu_0.tolist() 
    
    return hu
