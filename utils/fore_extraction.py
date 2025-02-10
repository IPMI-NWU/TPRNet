import numpy as np
import cv2
import math
from monai.transforms import CropForeground
def getForeCoord(image_tensor):
    crop_foreground = CropForeground(return_coords=True)
    _,start_coord,end_coord= crop_foreground(image_tensor)
    return start_coord,end_coord
#    length can change
#def getSquare(start,end):
#    height=end[1]-start[1]
#    width=end[2]-start[2]
#    area=height*width
#    length=int(math.sqrt(area))
#    length=length+1 if length%2==1 else length
#    mid_x=(end[1]+start[1])/2
#    mid_y=(end[2]-start[2])/2
#    start_x=int(mid_x-(length/2))
#    start_y=int(mid_y-(length/2))
#    if start_x<0:
#        start_x=0
#    if start_y<0:
#        start_y=0
#    return (start_x,start_y),length
    
def getSquare(start,end,square_size):
    height=end[0]-start[0]
    width=end[1]-start[1]
    length=square_size
    mid_x=(end[0]+start[0])/2
    mid_y=(end[1]-start[1])/2
    start_x=int(mid_x-length/2)
    start_y=int(mid_y-length/2)
    if start_x<0:
        start_x=0
    if start_y<0:
        start_y=0
    if start_x>512-length:
        start_x=512-length 
    if start_y>512-length:
        start_y=512-length
    return (start_x,start_y),length