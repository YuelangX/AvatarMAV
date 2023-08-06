import os
from yacs.config import CfgNode as CN
 

class config_base():

    def __init__(self):
        self.cfg = CN()
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()


class config_headnerf(config_base):

    def __init__(self):
        super(config_headnerf, self).__init__()

        self.cfg.start_epoch = 0
        self.cfg.gpu_id = 0
        self.cfg.load_headnerf_checkpoint = ''
        self.cfg.lr_feat_vol = 0.0
        self.cfg.lr_feat_net = 0.0
        self.cfg.lr_deform_vol = 0.0
        self.cfg.lr_deform_net = 0.0
        self.cfg.batch_size = 1
        

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''
        self.cfg.dataset.video_name = ''
        self.cfg.dataset.resolution = 512
        
        self.cfg.headmodule = CN()
        self.cfg.headmodule.deform_bs_res = 32
        self.cfg.headmodule.deform_bs_dim = 3
        self.cfg.headmodule.deform_linear = [54, 128, 3]
        self.cfg.headmodule.feature_res = 64
        self.cfg.headmodule.feature_dim = 4
        self.cfg.headmodule.density_linear = [140, 128, 1]
        self.cfg.headmodule.color_linear = [167, 128, 1]
        self.cfg.headmodule.interp_level = 3
        self.cfg.headmodule.exp_dim = 32
        self.cfg.headmodule.embedding_freq = 4
        self.cfg.headmodule.deform_bbox = [[-1.2, 1.2], [-1.1, 1.6], [-1.8, 0.9]]
        self.cfg.headmodule.feature_bbox = [[-1.2, 1.2], [-1.1, 1.6], [-1.8, 0.9]]
        self.cfg.headmodule.noise = 0.0
        

        self.cfg.neuralcamera = CN()
        self.cfg.neuralcamera.model_bbox = [[-1.2, 1.2], [-1.1, 1.6], [-1.8, 0.9]]
        self.cfg.neuralcamera.image_size = 256
        self.cfg.neuralcamera.max_samples = 2048
        self.cfg.neuralcamera.N_samples = 16
        self.cfg.neuralcamera.N_importance = 16
        self.cfg.neuralcamera.near_far = [0.0, 1.0]

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''
        self.cfg.recorder.logdir = ''
        self.cfg.recorder.checkpoint_path = ''
        self.cfg.recorder.result_path = ''
        self.cfg.recorder.save_freq = 1
        self.cfg.recorder.show_freq = 1

