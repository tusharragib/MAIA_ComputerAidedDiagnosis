#Importing Libraries.
import time
import os
import pandas as pd
import cv2
import cv2 as cv

#Importing other functions.
import hsv_features as hf  #For extracting the hsv features.
import grayrgb_mean as grgbm  #For extracting the graylevel and RGB channels' mean features.
import glcm  #For extracting the GLCM features.
import haralick_features as harf  #For extracting the haralick features.
import hu_moments as hu  #For extracting the hu moment features.
import lbp_features as lf  #For extracting the local binary pattern features.
import whiteBalance as wb  #For applying the white balance correction on the images.
import gabor_features as gf  #For extracting the gabor features.

#Initiating the starting time.
start_time = time.time()

#Creating the pathfolders to perform the batch operation for Feature extractions.
pathFolder = "/media/tushar/HDD/Academic/Semester_3/CAD_Computer Aided Diagnosis_5_Credits/Lab/Challenge-1 Data/val"#the pathfolder for Challenge-1 validation data.
folder_les = pathFolder + '/les/'
folder_nv = pathFolder + '/nv/'
filesArrayLes = [x for x in os.listdir(folder_les) if os.path.isfile(os.path.join(folder_les,x))]
filesArrayNv = [x for x in os.listdir(folder_nv) if os.path.isfile(os.path.join(folder_nv,x))]

#Creating an empty dataframe to store the feature values.
df = pd.DataFrame({
                   "name":[], 
                   "class":[], 
                   
                   "grayScale_mean":[], "blueChannel_mean":[], "greenChannel_mean":[], "redChannel_mean":[], 

                   "HSV_hue_mean":[], "HSV_saturation_mean":[], "HSV_value_mean":[],
                   "HSV_hue_std":[], "HSV_saturation_std":[], "HSV_value_std":[],
                   "HSV_hue_thirdMoment":[], "HSV_saturation_thirdMoment":[], "HSV_value_thirdMoment":[],
                   
                   "GLCM_Energy": [], "GLCM_Correlation": [], "GLCM_Homogeneity": [], "GLCM_Contrast": [], "GLCM_Dissimilarity": [],
                   
                   "Haralick_1":[], "Haralick_2":[], "Haralick_3":[], "Haralick_4":[], "Haralick_5":[],
                   "Haralick_6":[], "Haralick_7":[], "Haralick_8":[], "Haralick_9":[], "Haralick_10":[],
                   "Haralick_11":[], "Haralick_12":[], "Haralick_13":[],
                   
                   "hu_1":[], "hu_2":[], "hu_3":[], "hu_4":[],
                   "hu_5":[], "hu_6":[], "hu_7":[],
                   
                   "lbp_1":[], "lbp_2":[], "lbp_3":[], "lbp_4":[], "lbp_5":[],
                   "lbp_6":[], "lbp_7":[], "lbp_8":[], "lbp_9":[], "lbp_10":[],
                   "lbp_11":[], "lbp_12":[], "lbp_13":[], "lbp_14":[], "lbp_15":[],
                   "lbp_16":[], "lbp_17":[], "lbp_18":[], "lbp_19":[], "lbp_20":[],
                   "lbp_21":[], "lbp_22":[], "lbp_23":[], "lbp_24":[], "lbp_25":[],
                   "lbp_26":[], "lbp_27":[],
                   
                   "gabor_1":[], "gabor_2":[], "gabor_3":[], "gabor_4":[], "gabor_5":[],
                   "gabor_6":[], "gabor_7":[], "gabor_8":[]
                   })

#Extracting the features and creating dataframe for all the les images.
for file_name in filesArrayLes:
    file_name_no_extension = os.path.splitext(file_name)[0]
    image = cv.imread(folder_les+file_name)

    img = wb.white_balance_loops(image)
    
    hsv_features = hf.hsv_color_moments(img)
    grgb_mean_features = grgbm.gray_rgb_mean(img) 
    glcm_features = glcm.glcmFeatures(img)
    haralick_features = harf.haralick_features(img)
    hu_moments_features = hu.hu_moments(img)
    lbp_features = lf.lbp_features(img)
    gabor_features = gf.gabor_features(img)
    
    
    df = df.append({
                    "name": file_name, 
                    "class": 1, 
                    
                    "grayScale_mean": grgb_mean_features[0], "blueChannel_mean": grgb_mean_features[1], "greenChannel_mean": grgb_mean_features[2],"redChannel_mean": grgb_mean_features[3],
                    
                    "HSV_hue_mean": hsv_features[0], "HSV_saturation_mean": hsv_features[1], "HSV_value_mean": hsv_features[2],
                    "HSV_hue_std": hsv_features[3], "HSV_saturation_std": hsv_features[4], "HSV_value_std": hsv_features[5],
                    "HSV_hue_thirdMoment": hsv_features[6], "HSV_saturation_thirdMoment": hsv_features[7], "HSV_value_thirdMoment": hsv_features[8],
                    
                    "GLCM_Energy": glcm_features[0], "GLCM_Correlation": glcm_features[1], "GLCM_Homogeneity": glcm_features[2], 
                    "GLCM_Contrast": glcm_features[3], "GLCM_Dissimilarity": glcm_features[4],
                    
                    "Haralick_1":haralick_features[0], "Haralick_2":haralick_features[1], "Haralick_3":haralick_features[2], "Haralick_4":haralick_features[3], "Haralick_5":haralick_features[4],
                    "Haralick_6":haralick_features[5], "Haralick_7":haralick_features[6], "Haralick_8":haralick_features[7], "Haralick_9":haralick_features[8], "Haralick_10":haralick_features[9],
                    "Haralick_11":haralick_features[10], "Haralick_12":haralick_features[11], "Haralick_13":haralick_features[12],
                    
                    "hu_1":hu_moments_features[0], "hu_2":hu_moments_features[1], "hu_3":hu_moments_features[2], "hu_4":hu_moments_features[3],
                    "hu_5":hu_moments_features[4], "hu_6":hu_moments_features[5], "hu_7":hu_moments_features[6],
                    
                    "lbp_1":lbp_features[0], "lbp_2":lbp_features[1], "lbp_3":lbp_features[2], "lbp_4":lbp_features[3], "lbp_5":lbp_features[4],
                    "lbp_6":lbp_features[5], "lbp_7":lbp_features[6], "lbp_8":lbp_features[7], "lbp_9":lbp_features[8], "lbp_10":lbp_features[9],
                    "lbp_11":lbp_features[10], "lbp_12":lbp_features[11], "lbp_13":lbp_features[12], "lbp_14":lbp_features[13], "lbp_15":lbp_features[14],
                    "lbp_16":lbp_features[15], "lbp_17":lbp_features[16], "lbp_18":lbp_features[17], "lbp_19":lbp_features[18], "lbp_20":lbp_features[19],
                    "lbp_21":lbp_features[20], "lbp_22":lbp_features[21], "lbp_23":lbp_features[22], "lbp_24":lbp_features[23], "lbp_25":lbp_features[24],
                    "lbp_26":lbp_features[25], "lbp_27":lbp_features[26],
                    
                    "gabor_1":gabor_features[0], "gabor_2":gabor_features[1], "gabor_3":gabor_features[2], "gabor_4":gabor_features[3], "gabor_5":gabor_features[4],
                   "gabor_6":gabor_features[5], "gabor_7":gabor_features[6], "gabor_8":gabor_features[7]
                    } 
                   ,ignore_index = True)
    print(file_name)


#Extracting the features and creating dataframe for all the nv images.
for file_name in filesArrayNv:
    file_name_no_extension = os.path.splitext(file_name)[0]
    
    image = cv.imread(folder_nv+file_name)
    
    img = wb.white_balance_loops(image)
    
    hsv_features = hf.hsv_color_moments(img)
    grgb_mean_features = grgbm.gray_rgb_mean(img) 
    glcm_features = glcm.glcmFeatures(img)
    haralick_features = harf.haralick_features(img)
    hu_moments_features = hu.hu_moments(img)
    lbp_features = lf.lbp_features(img)
    
    df = df.append({
                    "name": file_name, 
                    "class": 0, 
                    
                    "grayScale_mean": grgb_mean_features[0], "blueChannel_mean": grgb_mean_features[1], "greenChannel_mean": grgb_mean_features[2],"redChannel_mean": grgb_mean_features[3],
                    
                    "HSV_hue_mean": hsv_features[0], "HSV_saturation_mean": hsv_features[1], "HSV_value_mean": hsv_features[2],
                    "HSV_hue_std": hsv_features[3], "HSV_saturation_std": hsv_features[4], "HSV_value_std": hsv_features[5],
                    "HSV_hue_thirdMoment": hsv_features[6], "HSV_saturation_thirdMoment": hsv_features[7], "HSV_value_thirdMoment": hsv_features[8],
                    
                    "GLCM_Energy": glcm_features[0], "GLCM_Correlation": glcm_features[1], "GLCM_Homogeneity": glcm_features[2], 
                    "GLCM_Contrast": glcm_features[3], "GLCM_Dissimilarity": glcm_features[4],
                    
                    "Haralick_1":haralick_features[0], "Haralick_2":haralick_features[1], "Haralick_3":haralick_features[2], "Haralick_4":haralick_features[3], "Haralick_5":haralick_features[4],
                    "Haralick_6":haralick_features[5], "Haralick_7":haralick_features[6], "Haralick_8":haralick_features[7], "Haralick_9":haralick_features[8], "Haralick_10":haralick_features[9],
                    "Haralick_11":haralick_features[10], "Haralick_12":haralick_features[11], "Haralick_13":haralick_features[12],
                    
                    "hu_1":hu_moments_features[0], "hu_2":hu_moments_features[1], "hu_3":hu_moments_features[2], "hu_4":hu_moments_features[3],
                    "hu_5":hu_moments_features[4], "hu_6":hu_moments_features[5], "hu_7":hu_moments_features[6],
                    
                    "lbp_1":lbp_features[0], "lbp_2":lbp_features[1], "lbp_3":lbp_features[2], "lbp_4":lbp_features[3], "lbp_5":lbp_features[4],
                    "lbp_6":lbp_features[5], "lbp_7":lbp_features[6], "lbp_8":lbp_features[7], "lbp_9":lbp_features[8], "lbp_10":lbp_features[9],
                    "lbp_11":lbp_features[10], "lbp_12":lbp_features[11], "lbp_13":lbp_features[12], "lbp_14":lbp_features[13], "lbp_15":lbp_features[14],
                    "lbp_16":lbp_features[15], "lbp_17":lbp_features[16], "lbp_18":lbp_features[17], "lbp_19":lbp_features[18], "lbp_20":lbp_features[19],
                    "lbp_21":lbp_features[20], "lbp_22":lbp_features[21], "lbp_23":lbp_features[22], "lbp_24":lbp_features[23], "lbp_25":lbp_features[24],
                    "lbp_26":lbp_features[25], "lbp_27":lbp_features[26],
                    
                    "gabor_1":gabor_features[0], "gabor_2":gabor_features[1], "gabor_3":gabor_features[2], "gabor_4":gabor_features[3], "gabor_5":gabor_features[4],
                   "gabor_6":gabor_features[5], "gabor_7":gabor_features[6], "gabor_8":gabor_features[7]
                    } 
                   ,ignore_index = True)
    print(file_name)

#Saving the dataframe as a CSV file.
df.to_csv('val_fetures_extraction_les_nv.csv', index = False)

#Printing the termination time.
print(time.time()-start_time)
