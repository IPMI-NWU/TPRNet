import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='con_ce':
            return self.conloss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        # loss = torch.mean(torch.sum(-target * torch.log(F.softmax(logit, dim=1)), dim=1))
        # loss = torch.mean(torch.sum(-target * nn.LogSoftmax()(logit), dim=1))
        #loss = nn.BCEWithLogitsLoss()(logit, target)
        loss = nn.BCELoss()(logit, target)
        return loss
        
#    def conloss(self, logit, target):
#        logit = torch.where(logit > 0, 1, 0)
#        shp = logit.shape
#
#        img_pad = torch.zeros([shp[0] + 4, shp[0] + 4])
#        img_pad[2:-2, 2:-2] = logit
#        dir_array = torch.zeros([shp[0], shp[1],5])
#        dir_array=dir_array.to(device='cuda')
#        for i in range(shp[0]):
#            for j in range(shp[1]):
#                if logit[i, j] == 0:
#                    continue
#                dir_array[i, j,0] = img_pad[i, j]
#                dir_array[i, j,1] = img_pad[i, j + 4]
#                dir_array[i, j,2] = img_pad[i + 2, j + 2]
#                dir_array[i, j,3] = img_pad[i + 4, j]
#                dir_array[i, j,4] = img_pad[i + 4, j + 4]
#        target=target.permute(1,2,0)
#        loss = nn.BCEWithLogitsLoss()(dir_array, target)
#        return loss
    
    def get_con(self,logit,d):
        device = logit.device
        batch_size = logit.shape[0]
        logit_binary = torch.where(logit > 0.5, 1, 0)
        shp = logit.shape[1:]
        board=d+1
        # 创建 img_pad，避免内存分配开销  
        img_pad = torch.zeros([batch_size, shp[0] + 2*board, shp[1] + 2*board], device=device)
        img_pad[:, board:-board, board:-board] = logit_binary

        # 利用张量操作替代循环
        i_indices, j_indices = torch.nonzero(logit_binary, as_tuple=True)[1:]
        dir_array = torch.zeros([batch_size, shp[0], shp[1], 5], device=device)
        dir_array[:, i_indices, j_indices, 0] = img_pad[:, i_indices, j_indices]
        dir_array[:, i_indices, j_indices, 1] = img_pad[:, i_indices, j_indices + 2*board]
        dir_array[:, i_indices, j_indices, 2] = img_pad[:, i_indices + board, j_indices + board]
        dir_array[:, i_indices, j_indices, 3] = img_pad[:, i_indices + 2*board, j_indices]
        dir_array[:, i_indices, j_indices, 4] = img_pad[:, i_indices + 2*board, j_indices + 2*board]
        dir_array=dir_array.permute(0, 2, 3, 1)
        return dir_array

    def conloss(self, logit, target, d):
        pre=self.get_con(logit,d)
        tar=self.get_con(target,d)
        # 确保 target 的维度在计算损失时正确
        loss = nn.BCELoss()(pre, tar)
        return loss

    
    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        #print(y_pred, y_true[:,:,256])
        y_pred=F.sigmoid(y_pred)
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_true)
        #print(a,b)
        return a+b
from .soft_skeleton import soft_skel  
def soft_dice(y_true, y_pred):
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1. - coeff

class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        

    def forward(self, y_pred,y_true ):
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
                    torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
                    torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice        

#class MultiScaleLoss(nn.Module):
#    def __init__(self, weight=None):
#        super(MultiScaleLoss, self).__init__()
#        self.weight = weight if weight is not None else [1.0] * 7
#        assert len(self.weight) == 7, "Weight list must contain 7 elements."
#        self.db=dice_bce_loss()
#
#    def get_m(self,r,c):
#        num_batch=r.shape[0]
#        height=min(r.shape[1],r.shape[2])
#        width=height
#        
#        c = torch.clamp(c, 0, height-1)
#        r = torch.clamp(r, 0, width-1)
#        print(1,r.requires_grad)
#        #print(num_batch, width, height)
#        count_matrix = torch.zeros((num_batch, width, height), dtype=int,device=torch.device('cuda'))
#        for i in range(num_batch):
#            # Flatten the c and r coordinates for the i-th sample
#            flat_c = c[i].reshape(-1).long()
#            flat_r = r[i].reshape(-1).long()
#            
#            # Calculate flattened indices
#            flat_indices = flat_c * width + flat_r
#            print(2,flat_indices.requires_grad)
#            # Calculate bincount for the flattened indices
#            bincount = torch.bincount(flat_indices, minlength=width*height)
#            
#            count_matrix[i] = bincount.reshape(width, height)
#
#        
#        max_count = count_matrix.max()
#        count_matrix = torch.where(count_matrix > (max_count / 2), 1, 0)
#        return count_matrix
#    def forward(self, m, target):
#        assert len(m) == 7, "Input list m must contain 7 elements."
#        losses = []
#        for i in range(7):
#            # Resize the mask to the target size
#            m1,m2=m[i]
#            r1,c1=m1
#            r2,c2=m2
#            
#            count_matrix1=self.get_m(r1,c1)
#            count_matrix2=self.get_m(r2,c2)
#            count_matrix=(count_matrix1+count_matrix2)/2
#            resized_mask = F.interpolate(count_matrix.unsqueeze(1), size=target.shape[2:], mode='bilinear', align_corners=False)
#            # Calculate loss
#            loss = F.mse_loss(resized_mask, target)
#            # Apply weight
#            weighted_loss = self.weight[i] * loss
#            losses.append(weighted_loss)
#        
#        # Sum all losses
#        total_loss = sum(losses)
#        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()

    def forward(self, predicted_mask, target_mask):
        # Flatten both predicted_mask and target_mask
        predicted_flat = predicted_mask.view(-1)
        target_flat = target_mask.view(-1)

        # Only consider elements where target_mask == 1
        valid_indices = (target_flat == 1).float()

        # Calculate binary cross entropy loss
        bce_loss = F.binary_cross_entropy(predicted_flat, target_flat, reduction='none')

        # Mask the loss to only include where target_mask == 1
        masked_loss = bce_loss * valid_indices

        # Take the mean only over the elements where target_mask == 1
        loss = torch.sum(masked_loss) / torch.sum(valid_indices)

        return loss

class MultiScaleLoss(nn.Module):
    def __init__(self, weight=None):
        super(MultiScaleLoss, self).__init__()
        self.weight = weight if weight is not None else [1.0] * 7
        assert len(self.weight) == 7, "Weight list must contain 7 elements."
        self.mloss=MaskedBCELoss()
    def get_m(self, r, c):
        batch_size = r.shape[0]
        a=min(r.shape[1],r.shape[2])
        image_size=[a,a]
        image_height, image_width = image_size
        device = r.device
        # Clamp the values and keep them as float for gradient computation

        density_map = torch.zeros((batch_size, image_height, image_width), dtype=torch.float32, device=device)
    
        # 计算 floor 和 ceil
        # 计算 floor 和 ceil
        x_floor = torch.floor(r).long().clamp(0, image_height - 1)
        y_floor = torch.floor(c).long().clamp(0, image_width - 1)
        x_ceil = torch.ceil(r).long().clamp(0, image_height - 1)
        y_ceil = torch.ceil(c).long().clamp(0, image_width - 1)
        
        x_weight = r - x_floor.float()
        y_weight = c - y_floor.float()
        
        # 计算权重
        weight_tl = (1 - x_weight) * (1 - y_weight)  # top-left
        weight_tr = x_weight * (1 - y_weight)        # top-right
        weight_bl = (1 - x_weight) * y_weight        # bottom-left
        weight_br = x_weight * y_weight              # bottom-right
        
        # 创建 batch 索引
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(r).flatten()
    
        # 展平坐标和权重
        x_floor = x_floor.flatten()
        y_floor = y_floor.flatten()
        x_ceil = x_ceil.flatten()
        y_ceil = y_ceil.flatten()
        
        weight_tl = weight_tl.flatten()
        weight_tr = weight_tr.flatten()
        weight_bl = weight_bl.flatten()
        weight_br = weight_br.flatten()
        
        # 计算散点索引
        idx_tl = (x_floor * image_width + y_floor).long()
        idx_tr = (x_ceil * image_width + y_floor).long()
        idx_bl = (x_floor * image_width + y_ceil).long()
        idx_br = (x_ceil * image_width + y_ceil).long()
        
        # 将权重分配给密度图
        density_map.view(batch_size, -1).scatter_add_(1, idx_tl.view(batch_size, -1), weight_tl.view(batch_size, -1))
        density_map.view(batch_size, -1).scatter_add_(1, idx_tr.view(batch_size, -1), weight_tr.view(batch_size, -1))
        density_map.view(batch_size, -1).scatter_add_(1, idx_bl.view(batch_size, -1), weight_bl.view(batch_size, -1))
        density_map.view(batch_size, -1).scatter_add_(1, idx_br.view(batch_size, -1), weight_br.view(batch_size, -1))
        
        return density_map.view(batch_size, image_height, image_width)
        
    def forward(self, m, target):
        assert len(m) == 7, "Input list m must contain 7 elements."
        losses = []
        
        for i in range(7):
            m3,m4 = m[i]
#            r1, c1 = m1
#            r2, c2 = m2
            r3, c3 = m3
            r4, c4 = m4
#            count_matrix1 = self.get_m(r1, c1)
#            count_matrix2 = self.get_m(r2, c2)
            count_matrix3 = self.get_m(r3, c3)
            count_matrix4 = self.get_m(r4, c4)
            count_matrix = count_matrix3 + count_matrix4
            ma=count_matrix.max()
            count_matrix=count_matrix/ma
            resized_mask = F.interpolate(count_matrix.unsqueeze(1), size=target.shape[2:], mode='bilinear', align_corners=False)
            loss = self.mloss(resized_mask, target)
            weighted_loss = self.weight[i] * loss
            losses.append(weighted_loss)
        
        total_loss = sum(losses)
        return total_loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




