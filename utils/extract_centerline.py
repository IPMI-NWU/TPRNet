import numpy as np
import cv2 as cv
np.set_printoptions(threshold=512*512+10)
def extract_centerline(img_bin):
    
    img_bin=img_bin[0]
    img_bin = np.array(img_bin,np.uint8)
    # print(img_bin.shape)
    # print(type(img_bin))
    # print(img_bin)
    ker_dilate = np.ones((3, 3), np.uint8)  # 膨胀核
    ker_erode = np.ones((4, 4), np.uint8)   # 腐蚀核

    img_bin = cv.dilate(img_bin, ker_dilate, iterations=1)  # 膨胀
    img_bin = cv.erode(img_bin, ker_erode, iterations=1)    # 腐蚀

    # 提取中心线
    skelet = cv.ximgproc.thinning(img_bin) 
    return skelet