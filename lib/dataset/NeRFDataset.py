import torch
import torchvision as tv
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class NeRFDataset(Dataset):

    def __init__(self, opt, eval=False, use_all=False):
        super(NeRFDataset, self).__init__()

        self.dataroot = opt.dataroot
        self.resolution = opt.resolution
        self.loader = tv.datasets.folder.default_loader
        self.transform = tv.transforms.Compose([tv.transforms.Resize(self.resolution), tv.transforms.ToTensor()])

        self.samples = []
        video_folder = os.path.join(self.dataroot, opt.video_name)
        N = len(glob.glob(os.path.join(video_folder, 'images', 'img_*')))
        for j in range(N):
            image_path = os.path.join(video_folder, 'images', 'img_%05d.jpg' % j)
            mask_path = os.path.join(video_folder, 'images', 'mask_%05d.jpg' % j)
            bfm_param_path = os.path.join(video_folder, 'params', 'params_%05d.npz' % j)
            bfm_param = np.load(bfm_param_path)
            pose = torch.from_numpy(bfm_param['pose'])
            scale = torch.from_numpy(bfm_param['scale'])
            exp = torch.from_numpy(bfm_param['exp_coeff'])
            sample = (image_path, mask_path, pose, scale, exp)
            self.samples.append(sample)
        

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        image_path = sample[0]
        image = self.transform(self.loader(image_path))
        mask_path = sample[1]
        if not os.path.exists(mask_path):
            mask = torch.ones_like(image)
        else:
            mask = self.transform(self.loader(mask_path))
        image = image * mask + torch.ones_like(image) * (1 - mask)

        pose = sample[2][0]
        scale = sample[3]
        exp = sample[4][0]

        index = torch.tensor(index).long()

        return {
                'image': image,
                'mask': mask,
                'pose': pose,
                'scale': scale,
                'exp': exp,
                'index': index}

    def __len__(self):
        return len(self.samples)
