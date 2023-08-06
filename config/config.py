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
        
        self.cfg.headnerfmodule = CN()
        self.cfg.headnerfmodule.deform_bs_res = 65
        self.cfg.headnerfmodule.deform_bs_dim = 3
        self.cfg.headnerfmodule.deform_linear = [81, 3]
        self.cfg.headnerfmodule.feature_res = 65
        self.cfg.headnerfmodule.feature_dim = 4
        self.cfg.headnerfmodule.density_linear = [108, 4]
        self.cfg.headnerfmodule.color_linear = [108, 4]
        self.cfg.headnerfmodule.interp_level = 3
        self.cfg.headnerfmodule.exp_dim = 32
        self.cfg.headnerfmodule.embedding_freq = 4
        self.cfg.headnerfmodule.deform_bbox = [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.1]]
        self.cfg.headnerfmodule.feature_bbox = [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.1]]
        self.cfg.headnerfmodule.noise = 0.0
        

        self.cfg.neuralcamera = CN()
        self.cfg.neuralcamera.model_bbox = [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.1]]
        self.cfg.neuralcamera.image_size = 512
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

