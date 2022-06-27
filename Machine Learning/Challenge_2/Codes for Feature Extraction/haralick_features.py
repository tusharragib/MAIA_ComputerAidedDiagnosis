# This code is used to extract 13 haralick features from each images. 

import cv2
import cv2 as cv 
import mahotas as mh

def haralick_features(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    image_har = mh.features.haralick(img_gray).mean(axis=0)
    
    img_har = image_har.tolist()
    
    return img_har
