import cv2
import numpy as np
import torch
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    np.random.seed(1)
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
def sample_centerline_and_crop(img, input_tensor, patch_size, step):
    # 读取二值图像（假设中心线为白色）
    input_tensor = input_tensor[0, 0]  # Assuming input_tensor is a batch, use the first image
    #print(input_tensor.device)

    # 找到中心线的所有白色像素（中心线部分为白色，背景为黑色）
    centerline_points = np.column_stack(np.where(img == 255))  # 获取所有白色像素的坐标
    #print(centerline_points)
    # 如果图像没有中心线（白色像素），返回空列表和空数组
    if centerline_points.shape[0] == 0:
        #print("没有找到中心线！")
        centerline_points.append(256,256)

    # 计算采样点的数量
    num_samples = int(centerline_points.shape[0] // step)
    if num_samples==0:
        num_samples=1
    sampled_points=farthest_point_sample(centerline_points,num_samples)
    cropped_images = []
    crop_coords = []
    
    # 沿中心线均匀采样
    for p in sampled_points:
        x, y = p  # 每隔 step 采样一个点

        # 确定裁剪区域的坐标
        top_left = (int(y - patch_size // 2), int(x - patch_size // 2))
        bottom_right = (int(y + patch_size // 2), int(x + patch_size // 2))
        if top_left[0] < 0:
            bottom_right = (bottom_right[0] - top_left[0], bottom_right[1])
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            bottom_right = (bottom_right[0], bottom_right[1] - top_left[1])
            top_left = (top_left[0], 0)
        if bottom_right[0] > img.shape[0]:
            top_left = (top_left[0] - (bottom_right[0] - img.shape[0]), top_left[1])
            bottom_right = (img.shape[0], bottom_right[1])
        if bottom_right[1] > img.shape[1]:
            top_left = (top_left[0], top_left[1] - (bottom_right[1] - img.shape[1]))
            bottom_right = (bottom_right[0], img.shape[1])
            
        # 裁剪图像
        cropped_img = input_tensor[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 将裁剪图像转化为 (1, 64, 64) 形状
        #print(padded_img.device)
        cropped_img=cropped_img.unsqueeze(0)
        cropped_images.append(cropped_img)
        crop_coords.append((top_left, bottom_right))
    # 将所有裁剪块合并为一个数组
    cropped_images = torch.concatenate(cropped_images, axis=0)
    cropped_images = cropped_images.unsqueeze(1)
    #print(len(crop_coords))
    return cropped_images, crop_coords
