from config.config import config_headnerf
import argparse
import torch
import os
from lib.utils.util_seed import seed_everything

from lib.dataset.NeRFDataset import NeRFDataset
from lib.module.HeadModule import HeadModule
from lib.module.NeuralCameraModule import NeuralCameraModule
from lib.recorder.Recorder import TrainRecorder
from lib.trainer.Trainer import Trainer

if __name__ == '__main__':
    seed_everything(2645647)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_xyl.yaml')
    arg = parser.parse_args()

    cfg = config_headnerf()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = NeRFDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    dataset_eval = NeRFDataset(cfg.dataset)

    device = torch.device('cuda:%d' % cfg.gpu_id)

    headnerf = HeadModule(cfg.headmodule).to(device)
    if os.path.exists(cfg.load_headnerf_checkpoint):
        headnerf.load_state_dict(torch.load(cfg.load_headnerf_checkpoint, map_location=lambda storage, loc: storage))

    neu_camera = NeuralCameraModule(headnerf, cfg.neuralcamera)
    optimizer = torch.optim.Adam([{'params' : headnerf.feature_volume, 'lr' : cfg.lr_feat_vol},
                                  {'params' : headnerf.density_linear.parameters(), 'lr' : cfg.lr_feat_net},
                                  {'params' : headnerf.color_linear.parameters(), 'lr' : cfg.lr_feat_net},
                                  {'params' : headnerf.deform_bs_volume, 'lr' : cfg.lr_deform_vol,},
                                  {'params' : headnerf.deform_mean_volume, 'lr' : cfg.lr_deform_vol,},
                                  {'params' : headnerf.deform_linear.parameters(), 'lr' : cfg.lr_deform_net}])
    recorder = TrainRecorder(cfg.recorder, dataset_eval)

    trainer = Trainer(dataloader, headnerf, neu_camera, optimizer, recorder, cfg.gpu_id)
    trainer.train(0, 100)
