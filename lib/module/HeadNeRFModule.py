import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder

class HeadNeRFModule(nn.Module):
    def __init__(self, opt):
        super(HeadNeRFModule, self).__init__()
        
        self.deform_bs_volume = nn.Parameter(torch.zeros([opt.exp_dim, opt.deform_bs_dim, opt.deform_bs_res, opt.deform_bs_res, opt.deform_bs_res]))
        self.deform_mean_volume = nn.Parameter(torch.zeros([opt.deform_bs_dim, opt.deform_bs_res, opt.deform_bs_res, opt.deform_bs_res]))
        self.deform_linear = MLP(opt.deform_linear)

        self.feature_volume = nn.Parameter(torch.zeros([opt.feature_dim, opt.feature_res, opt.feature_res, opt.feature_res]))
        self.density_linear = MLP(opt.density_linear)
        self.color_linear = MLP(opt.color_linear)

        self.interp_level = opt.interp_level
        self.exp_dim = opt.exp_dim

        self.feat_embedding, self.feat_out_dim = get_embedder(opt.embedding_freq)
        self.deform_embedding, self.deform_out_dim = get_embedder(opt.embedding_freq)
        self.view_embedding, self.view_out_dim = get_embedder(opt.embedding_freq)

        self.deform_bbox = opt.deform_bbox
        self.feature_bbox = opt.feature_bbox
        self.noise = opt.noise
        self.deform_scale = 0.1

    def query(self, data):
        B, C, N = data['query_pts'].shape
        query_pts = data['query_pts']
        query_viewdirs = data['query_viewdirs']
        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, 3:, None]
            S = data['scale'][:, :, None]
            query_pts = torch.bmm(R.permute(0,2,1), (query_pts - T)) / S
            query_viewdirs = torch.bmm(R.permute(0,2,1), query_viewdirs)

        exp = (data['exp'])[:, :self.exp_dim, None, None, None, None]
        deform_bs_volume = self.deform_bs_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)
        deform_mean_volume = self.deform_mean_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        #deform_volume = rearrange(exp * deform_bs_volume, 'b e c x y z -> b (e c) x y z') + deform_mean_volume
        #deform = self.mult_dist_interp(query_pts, deform_volume, self.deform_bbox)
        deform_volume = (exp * deform_bs_volume).sum(1) + deform_mean_volume
        deform = self.mult_dist_interp(query_pts, deform_volume, self.deform_bbox)
        #deform = self.interp(query_pts, deform_volume, self.deform_bbox)

        deform_embedding = self.deform_embedding(rearrange(deform, 'b c n -> (b n) c'))
        offset = rearrange(self.deform_linear(deform_embedding), '(b n) c -> b c n', b=B)
        #offset = deform
        deformed_pts = offset * self.deform_scale + query_pts
        feature = self.mult_dist_interp(deformed_pts, self.feature_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1), self.feature_bbox)
        #feature = self.mult_dist_interp(query_pts, self.feature_volume.unsqueeze(0).repeat(B, 1, 1, 1, 1), self.feature_bbox)

        feature_embedding = self.feat_embedding(rearrange(feature, 'b c n -> (b n) c'))
        query_viewdirs_embedding = self.view_embedding(rearrange(query_viewdirs, 'b c n -> (b n) c'))
        exp = rearrange((data['exp'] / 3)[:, :self.exp_dim, None].repeat(1, 1, N), 'b c n -> (b n) c')
        density = rearrange(self.density_linear(torch.cat([feature_embedding, exp], 1)), '(b n) c -> b c n', b=B)
        if self.training:
            density = density + torch.randn_like(density) * self.noise
        color = rearrange(self.color_linear(torch.cat([feature_embedding, query_viewdirs_embedding, exp], 1)), '(b n) c -> b c n', b=B)
        color = torch.sigmoid(color)
        #color = rearrange(self.deform_linear(deform_embedding), '(b n) c -> b c n', b=B)
        #color = torch.clamp(color * 5 + 0.5, 0, 1)
        
        data['offset'] = offset
        data['density'] = density
        data['color'] = color
        return data

    def interp(self, pts, volume, bbox):
        feature_volume = volume
        u = (pts[:, 0:1] - 0.5 * (bbox[0][0] + bbox[0][1])) / (0.5 * (bbox[0][1] - bbox[0][0]))
        v = (pts[:, 1:2] - 0.5 * (bbox[1][0] + bbox[1][1])) / (0.5 * (bbox[1][1] - bbox[1][0]))
        w = (pts[:, 2:3] - 0.5 * (bbox[2][0] + bbox[2][1])) / (0.5 * (bbox[2][1] - bbox[2][0]))
        uvw = rearrange(torch.cat([u, v, w], dim=1), 'b c (n t q) -> b n t q c', t=1, q=1)
        feature = torch.nn.functional.grid_sample(feature_volume, uvw)
        feature = rearrange(feature, 'b c n t q -> b c (n t q)')
        return feature

    def mult_dist_interp(self, pts, volume, bbox):
        u = (pts[:, 0:1] - 0.5 * (bbox[0][0] + bbox[0][1])) / (0.5 * (bbox[0][1] - bbox[0][0]))
        v = (pts[:, 1:2] - 0.5 * (bbox[1][0] + bbox[1][1])) / (0.5 * (bbox[1][1] - bbox[1][0]))
        w = (pts[:, 2:3] - 0.5 * (bbox[2][0] + bbox[2][1])) / (0.5 * (bbox[2][1] - bbox[2][0]))
        uvw = rearrange(torch.cat([u, v, w], dim=1), 'b c (n t q) -> b n t q c', t=1, q=1)

        feature_list = []
        for i in range(self.interp_level):
            feature_volume = volume[:,:,::2**i,::2**i,::2**i]
            feature = torch.nn.functional.grid_sample(feature_volume, uvw)
            feature = rearrange(feature, 'b c n t q -> b c (n t q)')
            feature_list.append(feature)
        feature = torch.cat(feature_list, dim=1)
        return feature

    def forward(self, data, forward_type='query'):
        if forward_type == 'query':
            data = self.query(data)
        return data
