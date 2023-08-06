import torch
import torch.nn.functional as F
from tqdm import tqdm


class Trainer():
    def __init__(self, dataloader, headnerf, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.headnerf = headnerf
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)

        self.intrinsic = torch.tensor([[2.5000e+03, 0.0000e+00, 1.2800e+02],
                                       [0.0000e+00, 2.5000e+03, 1.2800e+02],
                                       [0.0000e+00, 0.0000e+00, 1.0000e+00]]).float()
        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  4.0000]]).float()
        self.iter = 0
        self.lambda_reg = 1e-3
                                       
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):

                data['intrinsic'] = self.intrinsic
                data['extrinsic'] = self.extrinsic
                    
                to_cuda = ['image', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale', 'exp', 'index']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                data = self.camera(data, data['image'].shape[2])
                render_image = data['render_image']
                render_select = data['render_select']
                offset = data['offset']

                loss_rgb = F.l1_loss(render_image, data['image'] * render_select)
                loss_reg = torch.norm(offset, dim=1).mean()
                loss = loss_rgb + loss_reg * self.lambda_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter = idx + epoch * len(self.dataloader)
                log = {
                    'data': data,
                    'headnerf' : self.headnerf,
                    'camera' : self.camera,
                    'optim' : self.optimizer,
                    'loss_rgb' : loss_rgb,
                    'epoch' : epoch,
                    'iter' : self.iter
                }
                self.recorder.log(log)
