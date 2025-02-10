import numpy as np

class Evaluator:
    def __init__(self):
        self.dice_list = []

    def Dice_coefficient(self):
        return np.mean(self.dice_list)if self.dice_list else 0.0

    def _calculate_dice_batch(self, gt_batch, pred_batch):
        gt_batch=gt_batch.cpu().numpy()
        pred_batch=pred_batch.cpu().numpy()
        intersection = (gt_batch * pred_batch).sum(axis=(1, 2))
        union = gt_batch.sum(axis=(1, 2)) + pred_batch.sum(axis=(1, 2))
        dice_batch = 2.0 * intersection / union
        return dice_batch

    def add_batch(self, gt_image, pre_image):
        dice_batch = self._calculate_dice_batch(gt_image, pre_image)
        self.dice_list.append(dice_batch)

    def reset(self):
        self.dice_list = []