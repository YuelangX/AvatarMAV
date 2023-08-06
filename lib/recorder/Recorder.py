from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2
from PIL import Image


class TrainRecorder():
    def __init__(self, opt, dataset):
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = opt.name
        self.checkpoint_path = opt.checkpoint_path
        self.result_path = opt.result_path
        
        self.save_freq = opt.save_freq
        self.show_freq = opt.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

        self.dataset = dataset

    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['headnerf'].state_dict(), '%s/%s/headnerf_%d' % (self.checkpoint_path, self.name, log_data['iter']))

        if log_data['iter'] % self.show_freq == 0:

            i = (log_data['iter'] // self.show_freq)  - (log_data['iter'] // self.show_freq) // len(self.dataset) * len(self.dataset)
            data = self.dataset.get_item(i)
            to_cuda = ['image', 'mask', 'pose', 'scale', 'exp', 'index']
            for data_item in to_cuda:
                data[data_item] = data[data_item][None].to(device=log_data['data']['image'].device)
            data['intrinsic'] = log_data['data']['intrinsic']
            data['extrinsic'] = log_data['data']['extrinsic']
            with torch.no_grad():
                data = log_data['camera'].render(data, data['image'].shape[2])
            render_image = data['render_image']

            image = (data['image'][0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            render_image = (render_image[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            image = np.hstack((image, render_image))
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), image[:,:,::-1])
            