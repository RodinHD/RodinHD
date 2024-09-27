import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer
import raymarching

from gridencoder import GridEncoder


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 resolution=[128] * 3,
                 sigma_rank=[8] * 3,
                 color_rank=[24] * 3,
                 bg_resolution=[512, 512],
                 bg_rank=8,
                 color_feat_dim=24, # no use
                 num_layers=3,
                 hidden_dim=128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 triplane_channels=32,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution
 
        self.sigma_rank = sigma_rank
        self.color_rank = color_rank
 
        self.grid_sample = GridEncoder(input_dim=2, num_levels=3, level_dim=triplane_channels, per_level_scale=1, base_resolution=resolution[0], log2_hashmap_size=19, gridtype='tiled', align_corners=True)

 
        # render module (default to freq feat + freq dir)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, enc_dim = get_encoder('frequency', input_dim=color_rank[0], multires=2)
        self.encoder_sigma, enc_dim_sigma = get_encoder('frequency', input_dim=sigma_rank[0], multires=2)
        self.encoder_dir, enc_dim_dir = get_encoder('frequency', input_dim=3, multires=2)

        self.in_dim = enc_dim + enc_dim_dir + self.hidden_dim

        color_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3 # rgb
            else:
                out_dim = self.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        sigma_net = []
        sigma_net.append(nn.Linear(enc_dim_sigma, self.hidden_dim))
        sigma_net.append(nn.Softplus())
        sigma_net.append(nn.Linear(self.hidden_dim, 1+self.hidden_dim))
        self.sigma_net = nn.Sequential(*sigma_net)

        # background model
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            
            self.bg_resolution = bg_resolution
            self.bg_rank = bg_rank
            self.bg_mat = nn.Parameter(0.1 * torch.randn((1, bg_rank, bg_resolution[0], bg_resolution[1]))) # [1, R, H, W]
            
            bg_net =  []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = bg_rank + enc_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None
        
        self.plane_axes = self.generate_planes().cuda()
        self.inv_planes = torch.linalg.inv(self.plane_axes)
    
    def generate_planes(self):
        """
        Defines planes by the three vectors that form the "axes" of the
        plane. Should work with arbitrary number of planes and planes of
        arbitrary orientation.
        """
        return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]], dtype=torch.float32)

    def project_onto_planes(self, planes, coordinates, inv_planes=None):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        if inv_planes is not None:
            inv_planes = inv_planes.unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
        else:
            inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]
 
    def get_color_feat(self, triplane, x):
        mat_coord = self.project_onto_planes(self.plane_axes, x.unsqueeze(0), self.inv_planes)
        mat_feat =  torch.mean(self.grid_sample(triplane, mat_coord, self.bound), 0)
        return mat_feat

  
    # def forward(self, triplane, x, d):
    def forward_sample(self, triplane, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
       
        # rgb
        sampled_feat = self.get_color_feat(triplane, x)

        enc_color_feat = self.encoder(sampled_feat[:, :self.color_rank[0]])
        enc_sigma_feat = self.encoder_sigma(sampled_feat[:, self.color_rank[0]:])
        enc_d = self.encoder_dir(d)

        # sigma
        feat = self.sigma_net(enc_sigma_feat)
        sigma = trunc_exp(feat[:, 0])  
 
        h = torch.cat([enc_color_feat, feat[:, 1:], enc_d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb


    def density(self, triplane, x):
        # x: [N, 3], in [-bound, bound]
        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        sampled_feat = self.get_color_feat(triplane, x)
    
        enc_sigma_feat = self.encoder_sigma(sampled_feat[:,self.color_rank[0]:])

        # sigma
        feat = self.sigma_net(enc_sigma_feat)
        sigma = trunc_exp(feat[:, 0]) #F.softplus(sigma - 1.0)

        return {
            'sigma': sigma,
        }

    def color(self, triplane, x, d, mask=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # normalize to [-1, 1] inside aabb_train
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]

        # rgb
        sampled_feat = self.get_color_feat(triplane, x)

        enc_color_feat = self.encoder(sampled_feat[:, :self.color_rank[0]])
        enc_sigma_feat = self.encoder_sigma(sampled_feat[:, self.color_rank[0]:])
        enc_d = self.encoder_dir(d)

        # sigma
        feat = self.sigma_net(enc_sigma_feat)
        h = torch.cat([enc_color_feat, feat[:, 1:], enc_d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs

    def background(self, x, d):
        # x: [N, 2] in [-1, 1]

        N = x.shape[0]

        h = F.grid_sample(self.bg_mat, x.view(1, N, 1, 2), align_corners=True).view(-1, N).T.contiguous() # [R, N] --> [N, R]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

 

    # L1 penalty for loss
    def density_loss(self, triplane):
        loss = 0
        for i in range(len(triplane)):
            loss = loss + torch.mean(torch.abs(triplane[i]))
        return loss
    
    def tv_loss(self, triplane):
        loss = 0
        tvreg = TVLoss()
        for i in range(len(triplane)):
            loss = loss +  tvreg(triplane[i]) * 1e-2  
        return loss

    def dist_loss(self, triplane):
        initial_coordinates = torch.rand((1000, 3), device= triplane[0].device) * 2 - 1
        perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 0.004
        x = torch.cat([initial_coordinates, perturbed_coordinates], dim=0)

        sampled_feat = self.get_color_feat(triplane, x)
        enc_sigma_feat = self.encoder_sigma(sampled_feat[:, self.color_rank[0]:])

        # sigma
        sigma = self.sigma_net(enc_sigma_feat)
        sigma = trunc_exp(sigma[:, 0])
        # sigma = sigma.squeeze(1)
        sigma_initial = sigma[:sigma.shape[0]//2]   
        sigma_perturbed = sigma[sigma.shape[0]//2:]
                                                                                          
        loss = F.l1_loss(sigma_initial, sigma_perturbed) + torch.mean(torch.abs(sigma_initial)) * 0.001
        return loss

    def iwc_loss(self, iwc_state):
        decoder_state = {}
        decoder_state.update(self.named_parameters())
        fisher_state = iwc_state['fisher_state']
        optpar_state = iwc_state['optpar_state']
        loss = torch.tensor(0.0, device=list(decoder_state.values())[0].device)
        for fisher_mlp, optpar_mlp in zip(fisher_state, optpar_state):
            for name, param in decoder_state.items():
                if name in fisher_mlp.keys():
                    fisher = fisher_mlp[name].to(param.device)
                    optpar = optpar_mlp[name].to(param.device)
                    loss += (fisher * (optpar - param).pow(2)).sum()
        return loss
 
    # optimizer utils
    def get_params(self, lr1, lr2):
        params = [
            # {'params': self.triplane, 'lr': lr1}, 
            {'params': self.sigma_net.parameters(), 'lr': lr2},
            {'params': self.color_net.parameters(), 'lr': lr2},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.bg_mat, 'lr': lr1})
            params.append({'params': self.bg_net.parameters(), 'lr': lr2})
        return params
        
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
