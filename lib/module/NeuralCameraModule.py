import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

from pytorch3d.transforms import so3_exponential_map


class NeuralCameraModule(nn.Module):
    def __init__(self, model_module, opt):
        super(NeuralCameraModule, self).__init__()

        self.model_module = model_module
        self.model_bbox = opt.model_bbox
        self.image_size = opt.image_size
        self.max_samples = opt.max_samples
        self.N_samples = opt.N_samples
        self.N_importance = opt.N_importance
        self.near_far = opt.near_far

    @staticmethod
    def gen_part_rays(extrinsic, intrinsic, resolution, image_size):
         # resolution (width, height)
        rot = extrinsic[:3, :3].transpose(0, 1)
        trans = -torch.matmul(rot, extrinsic[:3, 3:])
        c2w = torch.cat((rot, trans.reshape(3, 1)), dim=1)

        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        res_w = resolution[0].int().item()
        res_h = resolution[1].int().item()
        W = image_size[0].int().item()
        H = image_size[1].int().item()
        i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, res_w, device=c2w.device), torch.linspace(0.5, H-0.5, res_h, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        # rays [B, C, H, W]
        return rearrange(rays_o, 'h w c -> c h w'), rearrange(rays_d, 'h w c -> c h w')

    @staticmethod
    def coords_select(image, coords):
        select_rays = image[:, coords[:, 1], coords[:, 0]]
        return select_rays

    @staticmethod
    def gen_near_far_fixed(near, far, samples, batch_size, device):
        nf = torch.zeros((batch_size, 2, samples), device=device)
        nf[:, 0, :] = near
        nf[:, 1, :] = far
        return nf

    def gen_near_far(self, rays_o, rays_d, R, T, S):
        """calculate intersections with 3d bounding box for batch"""
        B = rays_o.shape[0]
        rays_o_can = torch.bmm(R.permute(0,2,1), (rays_o - T)) / S
        rays_d_can = torch.bmm(R.permute(0,2,1), rays_d) / S
        bbox = torch.tensor(self.model_bbox, dtype=rays_o.dtype, device=rays_o.device)
        mask_in_box_batch = []
        near_batch = []
        far_batch = []
        for b in range(B):
            norm_d = torch.linalg.norm(rays_d_can[b], axis=0, keepdims=True)
            viewdir = rays_d_can[b] / norm_d
            viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
            viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
            tmin = (bbox[:, :1] - rays_o_can[b, :, :1]) / viewdir
            tmax = (bbox[:, 1:2] - rays_o_can[b, :, :1]) / viewdir
            t1 = torch.minimum(tmin, tmax)
            t2 = torch.maximum(tmin, tmax)
            near = torch.max(t1, 0)[0]
            far = torch.min(t2, 0)[0]
            mask_in_box = near < far
            mask_in_box_batch.append(mask_in_box)
            near_batch.append((near / norm_d[0]))
            far_batch.append((far / norm_d[0]))
        mask_in_box_batch = torch.stack(mask_in_box_batch)
        near_batch = torch.stack(near_batch)
        far_batch = torch.stack(far_batch)
        return near_batch, far_batch, mask_in_box_batch

    @staticmethod
    def sample_pdf(density, z_vals, rays_d, N_importance):
        r"""sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """
        bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        _, _, _, weights = NeuralCameraModule.integrate(density, z_vals, rays_d)
        weights = weights[..., 1:-1] + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], dtype=weights.dtype, device=weights.device)

        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        sample_z = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        sample_z, _ = torch.sort(sample_z, dim=-1)

        return sample_z

    @staticmethod
    def integrate(density, z_vals, rays_d, color=None, method='nerf'):
        '''Transforms module's predictions to semantically meaningful values.
        Args:
            density: [num_rays, num_samples along ray, 4]. Prediction from module.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            acc_map: [num_rays]. Sum of weights along each ray.
            depth_map: [num_rays]. Estimated distance to object.
        '''

        dists = (z_vals[...,1:] - z_vals[...,:-1]) * 1e2
        dists = torch.cat([dists, torch.ones(1, device=density.device).expand(dists[..., :1].shape) * 1e10], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        if method == 'nerf':
            alpha = 1 - torch.exp(-F.relu(density[...,0])*dists)
        elif method == 'unisurf':
            alpha = density[...,0]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=density.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        acc_map = torch.sum(weights, -1)
        depth_map = torch.sum(weights * z_vals, -1)

        if color == None:
            return None, acc_map, depth_map, weights
        rgb_map = torch.sum(weights[..., None] * color, -2)
        return rgb_map, acc_map, depth_map, weights

    @staticmethod
    def render_rays(data, model_module, N_samples=64, N_importance=64):
        B, C, N = data['rays_o'].shape

        rays_o = rearrange(data['rays_o'], 'b c n -> (b n) c')
        rays_d = rearrange(data['rays_d'], 'b c n -> (b n) c')
        N_rays = rays_o.shape[0]
        rays_nf = rearrange(data['rays_nf'], 'b c n -> (b n) c')

        near, far = rays_nf[...,:1], rays_nf[...,1:] # [-1,1]

        if N_samples > 0:
            t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device).unsqueeze(0)
            z_vals = near*(1-t_vals) + far*t_vals
            z_vals = z_vals.expand([N_rays, N_samples])
            
            # 采样点 coarse
            query_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B)
            query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
            query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_samples, 1), '(b n) s c -> b c (n s)', b=B)
            data['query_pts'] = query_pts
            data['query_viewdirs'] = query_viewdirs
            with torch.no_grad():
                data = model_module(data, 'query')
            density_coarse = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
            z_vals = NeuralCameraModule.sample_pdf(density_coarse, z_vals, rays_d, N_importance)
            z_vals_fine = z_vals.clone()
        else:
            t_vals = torch.linspace(0., 1., steps=N_importance, device=rays_o.device).unsqueeze(0)
            z_vals = near*(1-t_vals) + far*t_vals
            z_vals = z_vals.expand([N_rays, N_importance])
            z_vals_fine = z_vals.clone()

        # 采样点 fine
        query_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_importance, 3]
        query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B)
        query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_importance, 1), '(b n) s c -> b c (n s)', b=B)
        data['query_pts'] = query_pts
        data['query_viewdirs'] = query_viewdirs
        data = model_module(data, 'query')
        density_fine = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
        color_fine = rearrange(data['color'], 'b c (n s) -> (b n) s c', n=N)
        density_fine = torch.cat([density_fine, torch.ones([density_fine.shape[0], 1, density_fine.shape[2]], device=density_fine.device) * 1e8], 1)
        color_fine = torch.cat([color_fine, torch.ones([color_fine.shape[0], 1, color_fine.shape[2]], device=color_fine.device)], 1)
        z_vals_fine = torch.cat([z_vals_fine, torch.ones([z_vals_fine.shape[0], 1], device=z_vals_fine.device) * 1e8], 1)
        
        render_image, render_mask, render_depth, _ = NeuralCameraModule.integrate(density_fine, z_vals_fine, rays_d, color=color_fine, method='nerf')
        render_image = rearrange(render_image, '(b n) c -> b c n', b=B)
        render_mask = rearrange(render_mask, '(b n c) -> b c n', b=B, c=1)
        render_depth = rearrange(render_depth, '(b n c) -> b c n', b=B, c=1)
        
        data.update({'render_image': render_image, 'render_mask': render_mask, 'render_depth': render_depth})
        return data
    

    def forward(self, data, resolution):
        B = data['exp'].shape[0]
        H = W = resolution
        device = data['exp'].device

        rays_o_grid, rays_d_grid = self.gen_part_rays(data['extrinsic'], 
                                                      data['intrinsic'], 
                                                      torch.FloatTensor([H, W]), 
                                                      torch.FloatTensor([self.image_size, self.image_size]))

        init_mask = torch.ones([H, W], device=device)
        coords = torch.nonzero(init_mask > 0, as_tuple=False)
        coords = coords[:, torch.arange(1, -1, -1, device=device).long()]
        if coords.shape[0] > self.max_samples:
            select_inds = torch.from_numpy(np.random.choice(coords.shape[0], size=(self.max_samples), replace=False))
            coords = coords[select_inds]
        
        rays_o = self.coords_select(rays_o_grid, coords)[None].repeat(B, 1, 1)
        rays_d = self.coords_select(rays_d_grid, coords)[None].repeat(B, 1, 1)
        rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
        R = so3_exponential_map(data['pose'][:, :3])
        T = data['pose'][:, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
        S = data['scale'][:, :, None]
        rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
        for b in range(B):
            rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
            rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]

        render_data = {
            'exp': data['exp'],
            'pose': data['pose'],
            'scale': data['scale'],
            'rays_o': rays_o,
            'rays_d': rays_d,
            'coords': coords,
            'rays_nf': rays_nf
        }
        render_data = self.render_rays(render_data, self.model_module, N_samples=self.N_samples, N_importance=self.N_importance)

        render_image = torch.ones([B, 3, H, W], device=device)
        render_mask = torch.ones([B, 1, H, W], device=device)
        render_select = torch.ones([B, 1, H, W], device=device)
        render_image[:, :, coords[:, 1], coords[:, 0]] = render_data['render_image']
        render_mask[:, :, coords[:, 1], coords[:, 0]] = render_data['render_mask']
        render_select[:, :, coords[:, 1], coords[:, 0]] = 1.0
        data['render_image'] = render_image
        data['render_mask'] = render_mask
        data['render_select'] = render_select
        data['offset'] = render_data['offset']
        return data

    def render(self, data, resolution):
        H = W = resolution
        device = data['exp'].device

        rays_o_grid, rays_d_grid = self.gen_part_rays(data['extrinsic'], 
                                                      data['intrinsic'], 
                                                      torch.FloatTensor([H, W]), 
                                                      torch.FloatTensor([self.image_size, self.image_size]))

        init_mask = torch.ones([H, W], device=device)
        coords_list = torch.nonzero(init_mask > 0, as_tuple=False)
        coords_list = coords_list[:, torch.arange(1, -1, -1, device=device).long()]
        
        render_image = torch.ones([1, 3, H, W], device=device)
        render_mask = torch.ones([1, 1, H, W], device=device)
        for i in range(0, coords_list.shape[0], self.max_samples):
            coords = coords_list[i:i+self.max_samples, :]
            rays_o = self.coords_select(rays_o_grid, coords)[None]
            rays_d = self.coords_select(rays_d_grid, coords)[None]
            rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], 1, device)
            R = so3_exponential_map(data['pose'][0:1, :3])
            T = data['pose'][0:1, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
            S = data['scale'][0:1, :, None]
            rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
            rays_nf[0, 0, mask_in_box[0]] = rays_near_bbox[0, mask_in_box[0]]
            rays_nf[0, 1, mask_in_box[0]] = rays_far_bbox[0, mask_in_box[0]]
            render_data = {
                'exp': data['exp'],
                'pose': data['pose'],
                'scale': data['scale'],
                'rays_o': rays_o,
                'rays_d': rays_d,
                'coords': coords,
                'rays_nf': rays_nf
            }
            render_data = self.render_rays(render_data, self.model_module, N_samples=self.N_samples, N_importance=self.N_importance)

            render_image[0, :, coords[:, 1], coords[:, 0]] = render_data['render_image'][0]
            render_mask[0, :, coords[:, 1], coords[:, 0]] = render_data['render_mask'][0]

        data['render_image'] = render_image
        data['render_mask'] = render_mask
        return data