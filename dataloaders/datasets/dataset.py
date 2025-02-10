from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.transform_img_lab import transform_img_lab

class Segmentation(Dataset):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('arcade'),
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, split, 'images')
        self._label_dir = os.path.join(self._base_dir, split, 'gt')
        self.class_names=[0,1]

        self.args = args
        self.split=split

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for filename in os.listdir(self._image_dir):
            _image = os.path.join(self._image_dir, filename)
            _cat = os.path.join(self._label_dir, filename)
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(filename[:-4])
            self.images.append(_image)
            self.categories.append(_cat)


        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.tr_batch_size * self.args.tr_batch_size


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        if self.split == "train":
            return self.transform_tr(sample), self.im_ids[index]
        elif self.split == 'val':
            return self.transform_val(sample), self.im_ids[index]
        elif self.split == 'test':
            return self.transform_test(sample), self.im_ids[index]


    def _make_img_gt_point_pair(self, index):
        
        _img = Image.open(self.images[index]).convert('L')
        
        _target = Image.open(self.categories[index])
        

        return _img, _target

    def transform_tr(self, sample):

        # composed_transforms = transforms.Compose([
            # # tr.RandomRotate(180),
            # tr.RandomHorizontalFlip(),
            # # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # # tr.RandomGaussianBlur(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # tr.ToTensor()])

        return transform_img_lab(sample['image'],sample['label'])

    def transform_val(self, sample):

#         composed_transforms = transforms.Compose([
#             tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             tr.ToTensor()])

        return transform_img_lab(sample['image'],sample['label'],'val')

    def transform_test(self, sample):

        return transform_img_lab(sample['image'],sample['label'],'val')

        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512
    args.batch_size = 1
    args.dataset = 'DeepGlobe'

    data_train = Segmentation(args, split='train')

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='DeepGlobe')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


