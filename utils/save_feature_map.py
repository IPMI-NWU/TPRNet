from PIL import Image
import numpy as np
import torch
def save_map_2(x,name,channel):
    feature_map = torch.softmax(x, dim=1)
    feature_map = feature_map[:, 1].data.cpu().numpy()
    feature_map[feature_map>0.5]=255
    feature_map[feature_map <0.5] = 0
    feature_map=feature_map.astype(np.uint8)
    feature_map=feature_map[0]
    image = Image.fromarray(feature_map)  # 将 NumPy 数组转换为 PIL 图像对象
    image.save("img/"+name+".png")
def save_map_1(x,name,folder):

    x=x.cpu().numpy()
    x=x[0]
    x[x>0.5]=255
    x[x <0.5] = 0
    feature_map=x.astype(np.uint8)
    image = Image.fromarray(feature_map)  # 将 NumPy 数组转换为 PIL 图像对象
    image.save(folder+name+".png")
def save_map_img(x,name,channel):
    
    x=x.data.cpu().numpy()
    x=x[0,0]
    x=x*255
    feature_map=x.astype(np.uint8)
    image = Image.fromarray(feature_map)  # 将 NumPy 数组转换为 PIL 图像对象
    image.save("img/"+name+".png")