from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import torch_scatter

from components.unet3d import Abstract3DUNet, DoubleConv
from components.mlp import MLP, MLP_V2
from networks.resunet import SparseResUNet2
from networks.transformer import TransformerSiamese
from networks.pointnet import MiniPointNetfeat
from common.torch_util import to_numpy
from common.visualization_util import (
    get_vis_idxs, render_nocs_pair)
from components.gridding import VirtualGrid

import MinkowskiEngine as ME
import numpy as np


class VolumeTrackingFeatureAggregator(pl.LightningModule):
    def __init__(self,
                 nn_channels=(38, 64, 64),
                 batch_norm=True,
                 lower_corner=(0,0,0),
                 upper_corner=(1,1,1),
                 grid_shape=(32, 32, 32),
                 reduce_method='max',
                 include_point_feature=True,
                 use_gt_nocs_for_train=True,
                 use_mlp_v2=True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        if use_mlp_v2:
            self.local_nn2 = MLP_V2(nn_channels, batch_norm=batch_norm, transpose_input=True)
        else:
            self.local_nn2 = MLP(nn_channels, batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        self.use_gt_nocs_for_train = use_gt_nocs_for_train
    
    def forward(self, nocs_data, batch_size, is_train=False):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        include_point_feature = self.include_point_feature
        reduce_method = self.reduce_method

        sim_points_frame2 = nocs_data['sim_points_frame2']
        if is_train and self.use_gt_nocs_for_train:
            points_frame2 = nocs_data['pos_gt_frame2']
        else:
            if 'refined_pos_frame2' in nocs_data:
                points_frame2 = nocs_data['refined_pos_frame2']
            else:
                points_frame2 = nocs_data['pos_frame2']
        nocs_features_frame2 = nocs_data['x_frame2']
        batch_idx_frame2 = nocs_data['batch_frame2']
        device = points_frame2.device
        float_dtype = points_frame2.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner, 
            upper_corner=upper_corner, 
            grid_shape=grid_shape, 
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype)
        
        # get aggregation target index
        points_grid_idxs_frame2 = vg.get_points_grid_idxs(points_frame2, batch_idx=batch_idx_frame2)
        flat_idxs_frame2 = vg.flatten_idxs(points_grid_idxs_frame2, keepdim=False)

        # get features
        features_list_frame2 = [nocs_features_frame2]
        if include_point_feature:
            points_grid_points_frame2 = vg.idxs_to_points(points_grid_idxs_frame2)
            local_offset_frame2 = points_frame2 - points_grid_points_frame2
            features_list_frame2.append(local_offset_frame2)
            features_list_frame2.append(sim_points_frame2)

        features_frame2 = torch.cat(features_list_frame2, axis=-1)
        
        # per-point transform
        if self.local_nn2 is not None:
            features_frame2 = self.local_nn2(features_frame2)

        # scatter
        volume_feature_flat_frame2 = torch_scatter.scatter(
            src=features_frame2.T, index=flat_idxs_frame2, dim=-1,
            dim_size=vg.num_grids, reduce=reduce_method)

        # reshape to volume
        feature_size = volume_feature_flat_frame2.shape[0]
        volume_feature_all = volume_feature_flat_frame2.reshape(
            (feature_size, batch_size) + grid_shape).permute((1,0,2,3,4))
        return volume_feature_all


class UNet3DTracking(pl.LightningModule):
    def __init__(self, in_channels, out_channels, f_maps=64, 
            layer_order='gcr', num_groups=8, num_levels=4):
        super().__init__()
        self.save_hyperparameters()
        self.abstract_3d_unet = Abstract3DUNet(
            in_channels=in_channels, out_channels=out_channels,
            final_sigmoid=False, basic_module=DoubleConv, f_maps=f_maps,
            layer_order=layer_order, num_groups=num_groups, 
            num_levels=num_levels, is_segmentation=False)
    
    def forward(self, data):
        result = self.abstract_3d_unet(data)
        return result


class ImplicitWNFDecoder(pl.LightningModule):
    def __init__(self,
            nn_channels=(128,256,256,3),
            batch_norm=True,
            last_layer_mlp=False,
            use_mlp_v2=False
            ):
        super().__init__()
        self.save_hyperparameters()
        if use_mlp_v2:
            self.mlp = MLP_V2(nn_channels, batch_norm=batch_norm, transpose_input=True)
        else:
            self.mlp = MLP(nn_channels, batch_norm=batch_norm, last_layer=last_layer_mlp)

    def forward(self, features_grid, query_points):
        """
        features_grid: (N,C,D,H,W)
        query_points: (N,M,3)
        """
        batch_size = features_grid.shape[0]
        if len(query_points.shape) == 2:
            query_points = query_points.view(batch_size, -1, 3)
        # normalize query points to (-1, 1), which is 
        # requried by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0
        # shape (N,C,M,1,1)
        sampled_features = F.grid_sample(
            input=features_grid, 
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1,1,3))), 
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (N,M,C)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0,2,1)
        
        # shape (N,M,C)
        out_features = self.mlp(sampled_features)

        return out_features


class VotingPoseEstimator:
    def __init__(self, num_axises):
        self.num_joints = num_axises
        self.heatmaps = None

    def set_heatmaps(self, heatmaps):
        self.heatmaps = heatmaps

    def vote(self, threshold=0.5):
        if self.heatmaps is None:
            raise ValueError("Heatmaps not set")

        max_vals = np.zeros((self.heatmaps.shape[0], self.num_joints))
        argmax_vals = np.zeros((self.heatmaps.shape[0], self.num_joints), dtype=np.int)

        for i in range(self.heatmaps.shape[0]):
            max_vals[i] = np.max(self.heatmaps[i], axis=(1, 2))
            argmax_vals[i] = np.argmax(self.heatmaps[i], axis=(1, 2))

        votes = np.zeros((self.heatmaps.shape[0], self.num_joints, 3))

        for i in range(self.heatmaps.shape[0]):
            for j in range(self.num_joints):
                if max_vals[i][j] >= threshold:
                    votes[i][j] = argmax_vals[i][j]

        return votes



class PointMeshNocsRefiner(pl.LightningModule):
    def __init__(self,
                 pc_pointnet_channels=(326, 256, 256, 1024),
                 mesh_pointnet_channels=(3, 64, 128, 1024),
                 pc_refine_mlp_channels=(2304, 1024, 512, 192),
                 mesh_refine_pointnet_channels=(2112, 512, 512, 1024),
                 mesh_refine_mlp_channels=(1024, 512, 256, 6),
                 detach_input_pc_feature=True,
                 detach_global_pc_feature=True,
                 detach_global_mesh_feature=True,
                 **kwargs):
        super(PointMeshNocsRefiner, self).__init__()
        self.pc_pointnet = MiniPointNetfeat(nn_channels=pc_pointnet_channels)
        self.mesh_pointnet = MiniPointNetfeat(nn_channels=mesh_pointnet_channels)
        self.mesh_refine_pointnet = MiniPointNetfeat(nn_channels=mesh_refine_pointnet_channels)
        self.pc_refine_mlp = MLP_V2(channels=pc_refine_mlp_channels, batch_norm=True)
        self.mesh_refine_mlp = MLP_V2(channels=mesh_refine_mlp_channels, batch_norm=True)
        self.detach_input_pc_feature = detach_input_pc_feature
        self.detach_global_pc_feature = detach_global_pc_feature
        self.detach_global_mesh_feature = detach_global_mesh_feature

    def forward(self, pc_nocs, pc_sim, pc_cls_logits, pc_feat, mesh_nocs, batch_size=16, **kwargs):
        """
        NOCS refiner to refine predicted PC NOCS
        :param pc_nocs: (B*N, 3)
        :param pc_sim: (B*N, 3)
        :param pc_cls_logits: (B*N, 64*3)
        :param pc_feat: (B*N, C)
        :param mesh_nocs: (B*M, 3)
        :param batch_size: int
        :return: refined PC NOCS logits (B*N, 64*3), refined mesh NOCS (B*M, 3)
        """
        pc_nocs = pc_nocs.view(batch_size, -1, 3)  # (B, N, 3)
        pc_sim = pc_sim.view(batch_size, -1, 3)  # (B, N, 3)
        mesh_nocs = mesh_nocs.view(batch_size, -1, 3)  # (B, M, 3)
        num_pc_points = pc_nocs.shape[1]  # N
        num_mesh_points = mesh_nocs.shape[1]  # M
        # detach logits
        pc_cls_logits = pc_cls_logits.view(batch_size, num_pc_points, -1).detach()  # (B, N, 64*3)
        pc_feat = pc_feat.view(batch_size, num_pc_points, -1)  # (B, N, C)
        if self.detach_input_pc_feature:
            pc_feat = pc_feat.detach()
        pc_input_all = torch.cat([pc_nocs, pc_sim, pc_cls_logits, pc_feat], dim=-1)  # (B, N, 3+3+64*3+C)
        pc_input_all = pc_input_all.transpose(1, 2)  # (B, 3+3+64*3+C, N)

        # pc pointnet
        pc_feat_dense, pc_feat_global = self.pc_pointnet(pc_input_all)  # (B, C', N), (B, C')

        # mesh pointnet
        mesh_input_all = mesh_nocs.transpose(1, 2)  # (B, 3, M)
        mesh_feat_dense, _ = self.mesh_pointnet(mesh_input_all)  # (B, C', M)
        pc_feat_global_expand = pc_feat_global.unsqueeze(2).expand(-1, -1, num_mesh_points)  # (B, C', M)
        if self.detach_global_pc_feature:
            pc_feat_global_expand = pc_feat_global_expand.detach()
        mesh_feat_dense_cat = torch.cat([mesh_feat_dense, pc_feat_global_expand], dim=1)  # (B, C'+C', M)
        _, mesh_refine_feat_global = self.mesh_refine_pointnet(mesh_feat_dense_cat)  # (B, C')

        # refine pc-nocs
        mesh_refine_feat_global_expand = mesh_refine_feat_global.unsqueeze(2).expand(-1,  -1, num_pc_points)  # (B, C', N)
        if self.detach_global_mesh_feature:
            mesh_refine_feat_global_expand = mesh_refine_feat_global_expand.detach()
        pc_feat_cat = torch.cat([pc_feat_dense, mesh_refine_feat_global_expand], dim=1)  # (B, C'+C', N)
        delta_pc_cls_logits = self.pc_refine_mlp(pc_feat_cat)  # (B, 64*3, N)
        delta_pc_cls_logits = delta_pc_cls_logits.transpose(1, 2)  # (B, N, 64*3)
        assert delta_pc_cls_logits.shape[-1] == pc_cls_logits.shape[-1]
        refine_pc_cls_logits = pc_cls_logits + delta_pc_cls_logits  # (B, N, 64*3)
        refine_pc_cls_logits = refine_pc_cls_logits.reshape(batch_size * num_pc_points, -1)  # (B*N, 64*3)

        # refine mesh-nocs
        mesh_refine_feat_global = mesh_refine_feat_global.unsqueeze(2)  # (B, C', 1)
        mesh_nocs_delta_logits = self.mesh_refine_mlp(mesh_refine_feat_global)  # (B, 6, 1)
        refine_offset, refine_scale = mesh_nocs_delta_logits[:, :3, 0].unsqueeze(1), \
                                      mesh_nocs_delta_logits[:, 3:, 0].unsqueeze(1)  # (B, 1, 3)
        nocs_center = torch.tensor([[[0.5, 0.5, 0.5]]]).to(mesh_nocs.get_device())  # (1, 1, 3)
        refined_mesh_nocs = (mesh_nocs - nocs_center) * refine_scale + nocs_center + refine_offset  # (B, M, 3)
        refined_mesh_nocs = refined_mesh_nocs.reshape(batch_size * num_mesh_points, -1)  # (B*M, 3)
        return refine_pc_cls_logits, refined_mesh_nocs

class FPS(pl.LightningModule):
    def __init__(self, output_k, **kwargs):
        super().__init__()
        self.k = output_k
    
    def calc_distances_forward(self, p0, points):
        return ((p0-points)**2).sum(axis=-1)

    def forward(self, pts, dim=3):
        pts = pts['out_features']
        batch_size = pts.shape[0]
        farthest_pts = torch.zeros((batch_size, self.k, dim)).to(pts.get_device())
        output_dict = dict(out_features=farthest_pts)
        max_vals, _ = torch.max(pts, dim=1, keepdim=True)
        min_vals, _ = torch.min(pts, dim=1, keepdim=True)
        init_point = (max_vals + min_vals) / 2
        distances = self.calc_distances_forward(init_point, pts)
        for i in range(0, self.k):
            argmax_distance = torch.argmax(distances, dim=1)
            farthest_pts[:, i, :] = pts[torch.arange(batch_size), argmax_distance]
            distances = torch.minimum(distances, self.calc_distances_forward(farthest_pts[:, i].unsqueeze(1), pts))
        return output_dict

class FPSLearnableLinear(pl.LightningModule):
    def __init__(self, num_surface_sample, output_k, layers_num=3, **kwargs):
        super().__init__()
        self.k = output_k
        minus = (num_surface_sample - output_k) // layers_num
        self.module_list = nn.ModuleList()
        for i in range(layers_num - 1):
            self.module_list.append(nn.Sequential(nn.Linear(num_surface_sample * 3, num_surface_sample * 3 - minus * 3), nn.BatchNorm1d(num_surface_sample * 3 - minus * 3), nn.ReLU()))
            num_surface_sample -= minus
        self.module_list.append(nn.Sequential(nn.Linear(num_surface_sample * 3, output_k * 3)))
        self.module_list = nn.Sequential(*self.module_list)

    def forward(self, pts):
        x = pts['out_features']
        output_dict = dict(out_features=None)
        x = x.contiguous().view(x.size(0), -1)
        x = self.module_list(x)
        x = x.view(x.size(0), -1, 3)
        output_dict['out_features'] = x
        return output_dict


class GDM(nn.Module):  # Garment deformation module
    def __init__(self, input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5,
                add_tps=True, add_depth=True, add_segmt=True, norm_layer=nn.InstanceNorm2d, use_dropout=False, device='cpu'):
        super(GDM, self).__init__()
        self.add_tps = add_tps
        self.add_depth = add_depth
        self.add_segmt = add_segmt

        self.extractionA = FeatureExtraction(input_nc_A, ngf, n_layers, norm_layer, use_dropout)
        self.extractionB = FeatureExtraction(input_nc_B, ngf, n_layers, norm_layer, use_dropout)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression_tps = FeatureRegression(input_nc=640, output_dim=2*grid_size**2)
        self.tps_grid_gen = TpsGridGen(img_height, img_width, grid_size=grid_size, device=device)

        if self.add_segmt:
            self.segmt_dec = SegmtDec()

        if self.add_depth:
            self.depth_dec = DepthDec(in_nc=1024)

    def forward(self, inputA, inputB):
        """
            input A: agnostic (batch_size,12,512,320)
            input B: flat cloth mask(batch_size,1,512,320)
        """
        output = {'theta_tps':None, 'grid_tps':None, 'depth':None, 'segmt':None}
        featureA = self.extractionA(inputA) # featureA: size (batch_size,512,32,20)
        featureB = self.extractionB(inputB) # featureB: size (batch_size,512,32,20)
        if self.add_depth or self.add_segmt:
            featureAB = torch.cat([featureA, featureB], 1) # input for DepthDec and SegmtDec: (batch_size,1024,32,20)
            if self.add_depth:
                depth_pred = self.depth_dec(featureAB)
                output['depth'] = depth_pred
            if self.add_segmt:
                segmt_pred = self.segmt_dec(featureAB)
                output['segmt'] = segmt_pred
        if self.add_tps:
            featureA = self.l2norm(featureA)
            featureB = self.l2norm(featureB)
            correlationAB = self.correlation(featureA, featureB) # correlationAB: size (batch_size, 640, 32, 32)
            theta_tps = self.regression_tps(correlationAB)
            grid_tps = self.tps_grid_gen(theta_tps)
            output['theta_tps'], output['grid_tps'] = theta_tps, grid_tps

        return output


class DRM(nn.Module):  # Depth Reconstruction Module
    def __init__(self, in_channel, out_channel, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(DRM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            norm_layer(self.ngf * 16)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


class TpsGridGen(pl.LightningModule):
    def __init__(self, out_h=512, out_w=320, use_regular_grid=True, grid_size=5):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32) # (512,320,3)
        # # sampling grid using meshgrid
        # self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # # grid_X,grid_Y: size [1,H,W,1]
        # self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3) # (1,512,320,1)
        # self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3) # (1,512,320,1)

        # # initialize regular grid for control points P_i
        # if use_regular_grid:
        #     axis_coords = np.linspace(-1, 1, grid_size)
        #     self.N = grid_size*grid_size # 25 control points
        #     P_Y, P_X = np.meshgrid(axis_coords,axis_coords) # BUG: should return (P_X, P_Y)?
        #     # P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
        #     P_X = np.reshape(P_X, (-1, 1)) # size (N=25,1)
        #     P_Y = np.reshape(P_Y, (-1, 1)) # size (N=25,1)
        #     P_X = torch.FloatTensor(P_X)
        #     P_Y = torch.FloatTensor(P_Y)
        #     self.P_X_base = P_X.clone() # size (N=25,1)
        #     self.P_Y_base = P_Y.clone() # size (N=25,1)
        #     self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0) # (1,N+3=28,N+3=28)
        #     self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
        #     self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
        self.grid = torch.zeros([self.out_h, self.out_w, 3], dtype=torch.float32)
        
        # sampling grid using meshgrid
        self.grid_X, self.grid_Y = torch.meshgrid(torch.linspace(-1, 1, out_w), torch.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1]
        self.grid_X = self.grid_X.unsqueeze(0).unsqueeze(3)
        self.grid_Y = self.grid_Y.unsqueeze(0).unsqueeze(3)
        
        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = torch.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size  # 25 control points
            P_Y, P_X = torch.meshgrid(axis_coords, axis_coords)  # Assuming the "BUG" is right
            P_X = P_X.contiguous().view(-1, 1)
            P_Y = P_Y.contiguous().view(-1, 1)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
        self.device_flag = 0
            
    def forward(self, theta):
        # theta.size(): (batch_size, N*2=50)
        if self.device_flag == 0:
            self.P_X_base = self.P_X_base.to(theta.get_device())
            self.P_Y_base = self.P_Y_base.to(theta.get_device())
            self.Li = self.Li.to(theta.get_device())
            self.P_X = self.P_X.to(theta.get_device())
            self.P_Y = self.P_Y.to(theta.get_device())
            self.grid_X = self.grid_X.to(theta.get_device())
            self.grid_Y = self.grid_Y.to(theta.get_device())
            self.grid = self.grid.to(theta.get_device())
            self.device_flag = 1
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3)) # (batch_size,512,512,2)
        
        return warped_grid
    
    def compute_L_inverse(self, X, Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        # a quick way to calculate distances between every control point pairs
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        # the TPS kernel funciont $U(r) = r^2*log(r)$
        # K.size: (N,N)
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared)) # BUG: should be torch.log(torch.sqrt(P_dist_squared))?
        # construct matrix L
        Z = torch.FloatTensor(N,1).fill_(1)
        O = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((Z,X,Y),1) # (N,3)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),O),1)),0) # (N+3,N+3)
        Li = torch.inverse(L) # (N+3,N+3)

        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3) # (batch_size, N*2=50, 1, 1)
        batch_size = theta.size()[0]
        # input are the corresponding control points P_i
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords.  
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # split theta into point coordinates (extract the displacements Q_X and Q_Y from theta)
        Q_X=theta[:,:self.N,:,:].squeeze(3) # (batch_size, N=25, 1)
        Q_Y=theta[:,self.N:,:,:].squeeze(3) # (batch_size, N=25, 1)
        # add the displacements to the original control points to get the target control points
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # compute weigths for non-linear part (multiply by the inverse matrix Li to get the coefficient vector W_X and W_Y)
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X) # (batch_size, N=25, 1)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y) # (batch_size, N=25, 1)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part (calculate the linear part $a_1 + a_x*a + a_y*y$)
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X) # (batch_size, 3, 1)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y) # (batch_size, 3, 1)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1) 
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points: size [1,H,W,2]
        # points_X_for_summation, points_Y_for_summation: size [1,H,W,1,N]
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X # (1,512,320,1,N=25)
            delta_Y = points_Y_for_summation-P_Y # (1,512,320,1,N=25)
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)  # (1,512,320,1,N=25)
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        # pass the distances to the radial basis function U
        # U: size [1,H,W,1,N]
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3) # (1,512,320,1)
        points_Y_batch = points[:,:,:,1].unsqueeze(3) # (1,512,320,1)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:]) # (batch_size,512,320,1)
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:]) # (batch_size,512,320,1)
        
        # points_X_prime, points_Y_prime: size [B,H,W,1]
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1], points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2], points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)), 4)
        
        # concatenate dense array points points_X_prime and points_Y_prime into a grid
        return torch.cat((points_X_prime, points_Y_prime), 3)

class FeatureL2Norm(pl.LightningModule):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()
        self.epsilon = 1e-6

    def forward(self, feature):
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+self.epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)


class FeatureCorrelation(pl.LightningModule):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        feature_A = feature_A.transpose(1, 2)
        correlation_tensor = torch.bmm(feature_B, feature_A)
        return correlation_tensor
    
class FeatureRegression(pl.LightningModule):
    def __init__(self, input_nc=640, output_dim=6):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(32768, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x) # (batch_size,64,8,5)
        x = x.reshape(x.size(0), -1) # (batch_size,2560)
        x = self.linear(x) # (batch_size,output_dim)
        x = self.tanh(x)
        return x

class TPS(pl.LightningModule):
    def __init__(self,
                 pc_mlp_channels=(128, 256, 256, 128),
                 batch_norm = True,
                 last_layer_mlp = False,
                 use_mlp_v2 = True,
                 mesh_pointnet_channels=(3, 64, 256, 1024),
                 grid_size = 64,
                 out_h = 512,
                 use_regular_grid = True,
                 out_w = 320,
                 input_nc = 4096):
        super().__init__()
        self.pc_mlpnet = ImplicitWNFDecoder(nn_channels=pc_mlp_channels, batch_norm=batch_norm, last_layer_mlp=last_layer_mlp, use_mlp_v2=use_mlp_v2)
        self.mesh_pointnet = MiniPointNetfeat(nn_channels=mesh_pointnet_channels)
        self.tps_grid_gen = TpsGridGen(out_h=out_h, out_w=out_w, use_regular_grid=use_regular_grid, grid_size=grid_size)
        self.correlation = FeatureCorrelation()
        self.regression_tps = FeatureRegression(input_nc=input_nc, output_dim=2*grid_size**2)
    
    def forward(self, surface_decoder_point, unet3d_result, encoder_result, batchsize):
        # if 'refined_pos_frame2' in encoder_result:
        #     points_frame2 = encoder_result['refined_pos_frame2'].reshape((batchsize, -1, 3)).transpose(1, 2)
        # else:
        #     points_frame2 = encoder_result['pos_frame2'].reshape((batchsize, -1, 3)).transpose(1, 2)
        mesh_nocs_pos_input = encoder_result['surf_query_points2'].reshape((batchsize, -1, 3))
        output = dict()
        feature_pc_fuse = self.pc_mlpnet(unet3d_result['out_feature_volume'], mesh_nocs_pos_input)
        feature_mesh = self.mesh_pointnet(mesh_nocs_pos_input.transpose(1, 2))[0]
        # feature_pc_fuse = F.normalize(feature_pc_fuse, p=2, dim=1)
        # feature_mesh = F.normalize(feature_mesh, p=2, dim=1)
        # correlationAB = self.correlation(feature_pc_fuse, feature_mesh)
        feature_pc_fuse_normalized = F.normalize(feature_pc_fuse.transpose(1, 2), p=2, dim=2)
        feature_mesh_normalized = F.normalize(feature_mesh.transpose(1, 2), p=2, dim=1)
        correlationAB = torch.bmm(feature_pc_fuse_normalized, feature_mesh_normalized)
        correlationAB = correlationAB.view((*correlationAB.shape[:-1], int(correlationAB.shape[-1]**0.5), int(correlationAB.shape[-1]**0.5)))
        theta_tps = self.regression_tps(correlationAB)
        grid_tps = self.tps_grid_gen(theta_tps)
        refined_tps_point = self.bilinear_interpolate(grid_tps, surface_decoder_point[..., [0, 2]])
        output['theta_tps'], output['grid_tps'], output['point_tps'] = theta_tps, grid_tps, refined_tps_point
        return output
    
    def bilinear_interpolate(self, grid, coords):
        """
        Performs bilinear interpolation for a batch of grid mappings and corresponding coordinates.
        
        Args:
        - grid (Tensor): The grid mapping of size (batch_size, H, W, 2).
        - coords (Tensor): The original coordinates of points of size (batch_size, num_points, 2).
        
        Returns:
        - Tensor: The interpolated coordinates of points of size (batch_size, num_points, 2).
        """
        # grid data range (-1, 1)
        # coords data range (0, 1)
        batch_size, H, W, _ = grid.shape
        coords_scaled = coords * torch.tensor([W - 1, H - 1], dtype=torch.float32).to(grid.get_device())
        x0 = torch.floor(coords_scaled[..., 0]).long().to(grid.get_device())
        x1 = x0 + 1
        y0 = torch.floor(coords_scaled[..., 1]).long().to(grid.get_device())
        y1 = y0 + 1
        x0 = torch.clamp(x0, 0, W-1)
        x1 = torch.clamp(x1, 0, W-1)
        y0 = torch.clamp(y0, 0, H-1)
        y1 = torch.clamp(y1, 0, H-1)
        tx = coords_scaled[..., 0] - x0
        ty = coords_scaled[..., 1] - y0
        wa = (1 - tx) * (1 - ty)
        wb = tx * (1 - ty)
        wc = (1 - tx) * ty
        wd = tx * ty
        Ia = grid[torch.arange(batch_size)[:, None], y0, x0]
        Ib = grid[torch.arange(batch_size)[:, None], y0, x1]
        Ic = grid[torch.arange(batch_size)[:, None], y1, x0]
        Id = grid[torch.arange(batch_size)[:, None], y1, x1]

        interpolated = wa.unsqueeze(-1) * Ia + wb.unsqueeze(-1) * Ib + wc.unsqueeze(-1) * Ic + wd.unsqueeze(-1) * Id
        return interpolated / 2 + 0.5

class ThetaLoss(pl.LightningModule):
    def __init__(self, grid_size=64):
        super(ThetaLoss, self).__init__()
        self.grid_size = grid_size
        
    def forward(self, theta):
        batch_size = theta.size()[0]
        coordinate = theta.view(batch_size, -1, 2) # (4,25,2)
        # coordinate+=torch.randn(coordinate.shape).cuda()/10
        row_loss = self.get_row_loss(coordinate, self.grid_size)
        col_loss = self.get_col_loss(coordinate, self.grid_size)
        # row_x, row_y, col_x, col_y: size [batch_size,15]
        row_x, row_y = row_loss[:,:,0], row_loss[:,:,1]
        col_x, col_y = col_loss[:,:,0], col_loss[:,:,1]
        # TODO: what does 0.08 mean?
        rx, ry, cx, cy = (torch.tensor([0.08]).to(theta.device) for i in range(4))
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()
        sec_diff_loss = rx_loss + ry_loss + cx_loss + cy_loss
        slope_loss = self.get_slope_loss(coordinate, self.grid_size).mean()

        theta_loss = sec_diff_loss + slope_loss

        return theta_loss
    
    def get_row_loss(self, coordinate, num):
        sec_diff = []
        for j in range(num):
            buffer = 0
            for i in range(num-1):
                # TODO: should be L2 distance according to ACGPN paper,  but not L1?
                diff = (coordinate[:, j*num+i+1, :]-coordinate[:, j*num+i, :]) ** 2
                if i >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff

        return torch.stack(sec_diff, dim=1)
    
    def get_col_loss(self, coordinate, num):
        sec_diff = []
        for i in range(num):
            buffer = 0
            for j in range(num - 1):
                # TODO: should be L2 distance according to ACGPN paper, but not L1?
                diff = (coordinate[:, (j+1)*num+i, :] - coordinate[:, j*num+i, :]) ** 2
                if j >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff
                
        return torch.stack(sec_diff,dim=1)
    
    def get_slope_loss(self, coordinate, num):
        slope_diff = []
        for j in range(num - 2):
            x, y = coordinate[:, (j+1)*num+1, 0], coordinate[:, (j+1)*num+1, 1]
            x0, y0 = coordinate[:, j*num+1, 0], coordinate[:, j*num+1, 1]
            x1, y1 = coordinate[:, (j+2)*num+1, 0], coordinate[:, (j+2)*num+1, 1]
            x2, y2 = coordinate[:, (j+1)*num, 0], coordinate[:, (j+1)*num, 1]
            x3, y3 = coordinate[:, (j+1)*num+2, 0], coordinate[:, (j+1)*num+2, 1]
            row_diff = torch.abs((y0 - y) * (x1 - x) - (y1 - y) * (x0 - x))
            col_diff = torch.abs((y2 - y) * (x3 - x) - (y3 - y) * (x2 -x))
            slope_diff.append(row_diff + col_diff)
            
        return torch.stack(slope_diff, dim=0)

class GridLoss(pl.LightningModule):
    def __init__(self, image_height, image_width, distance='l1'):
        super(GridLoss, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.distance = distance

    def forward(self, grid):
        gx = grid[:,:,:,0]
        gy = grid[:,:,:,1]
        gx_ctr = gx[:, 1:self.image_height-1, 1:self.image_width-1]
        gx_up = gx[:, 0:self.image_height-2, 1:self.image_width-1]
        gx_down = gx[:, 2:self.image_height, 1:self.image_width-1]
        gx_left = gx[:, 1:self.image_height-1, 0:self.image_width-2]
        gx_right = gx[:, 1:self.image_height-1, 2:self.image_width]

        gy_ctr = gy[:, 1:self.image_height-1, 1:self.image_width-1]
        gy_up = gy[:, 0:self.image_height-2, 1:self.image_width-1]
        gy_down = gy[:, 2:self.image_height, 1:self.image_width-1]
        gy_left = gy[:, 1:self.image_height-1, 0:self.image_width-2]
        gy_right = gy[:, 1:self.image_height-1, 2:self.image_width]

        if self.distance == 'l1':
            grid_loss_left = self._l1_distance(gx_left, gx_ctr)
            grid_loss_right = self._l1_distance(gx_right, gx_ctr)
            grid_loss_up = self._l1_distance(gy_up, gy_ctr)
            grid_loss_down = self._l1_distance(gy_down, gy_ctr)
        elif self.distance == 'l2':
            grid_loss_left = self._l2_distance(gx_left, gy_left, gx_ctr, gy_ctr)
            grid_loss_right = self._l2_distance(gx_right, gy_right, gx_ctr, gy_ctr)
            grid_loss_up = self._l2_distance(gx_up, gy_up, gx_ctr, gy_ctr)
            grid_loss_down = self._l2_distance(gx_down, gy_down, gx_ctr, gy_ctr)

        grid_loss = torch.sum(torch.abs(grid_loss_left-grid_loss_right) + torch.abs(grid_loss_up-grid_loss_down))

        return grid_loss
    
    def _l1_distance(self, x1, x2):
        return torch.abs(x1 - x2)
    
    def _l2_distance(self, x1, y1, x2, y2):
        return torch.sqrt(torch.mul(x1-x2, x1-x2) + torch.mul(y1-y2, y1-y2))


class GarmentTrackingPipeline(pl.LightningModule):
    """
    Use sparse ResUNet as backbone
    Use point-cloud pair as input(not mesh)
    Add transformer for self-attention and cross-attention
    """
    def __init__(self,
                 # sparse uned3d encoder params
                 sparse_unet3d_encoder_params,
                 # self-attention and cross-attention transformer params
                 transformer_params,
                 # pc nocs and mesh nocs refiner params
                 nocs_refiner_params,
                 # VolumeFeaturesAggregator params
                 volume_agg_params,
                 # unet3d params
                 unet3d_params,
                 # ImplicitWNFDecoder params
                 surface_decoder_params,
                 # added by zhangli and mingliangxu.
                 tps_params,
                 # add by zhangli and mingliangxu.
                 mlp_params,
                 use_mlp,
                 use_fps_branch,
                 use_tps,
                 # training params
                 learning_rate=1e-4,
                 optimizer_type='Adam',
                 loss_type='l2',
                 fps_branch_loss_weight=10.0,
                 volume_loss_weight=1.0,
                 warp_loss_weight=10.0,
                 tps_grid_loss_weight=1.0,
                 tps_theta_loss_weight=1.0,
                 tps_point_loss_weight=10,
                 nocs_loss_weight=1.0,
                 mesh_loss_weight=10.0,
                 use_nocs_refiner=True,
                 disable_pc_nocs_refine_in_test=False,
                 disable_mesh_nocs_refine_in_test=False,
                 # vis params
                 vis_per_items=0,
                 max_vis_per_epoch_train=0,
                 max_vis_per_epoch_val=0,
                 batch_size=None,
                 # debug params
                 debug=False
                 ):
        super().__init__()
        self.save_hyperparameters()

        criterion = None
        if loss_type == 'l2':
            criterion = nn.MSELoss(reduction='mean')
        elif loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise RuntimeError("Invalid loss_type: {}".format(loss_type))

        self.sparse_unet3d_encoder = SparseResUNet2(**sparse_unet3d_encoder_params)
        self.transformer_siamese = TransformerSiamese(**transformer_params)
        self.nocs_refiner = PointMeshNocsRefiner(**nocs_refiner_params)
        self.volume_agg = VolumeTrackingFeatureAggregator(**volume_agg_params)
        self.unet_3d = UNet3DTracking(**unet3d_params)
        self.surface_decoder = ImplicitWNFDecoder(**surface_decoder_params)
        if use_fps_branch:
            self.fps_like_surface_decoder = ImplicitWNFDecoder(**surface_decoder_params)
            self.last_fps_decoder = FPS(**mlp_params)
        if use_mlp:
            self.fps_forward = FPS(**mlp_params)
            self.fps_forward_learnable_linear = FPSLearnableLinear(**mlp_params)
        if use_tps:
            self.tps_module = TPS(**tps_params)
            self.tps_grid_loss_weight=tps_grid_loss_weight
            self.tps_theta_loss_weight=tps_theta_loss_weight
            self.tps_point_loss_weight=tps_point_loss_weight
            self.tps_grid_criterion = GridLoss(tps_params['out_h'], tps_params['out_w'], 'l2')
            self.tps_theta_criterion = ThetaLoss(tps_params['grid_size'])
            self.tps_point_criterion = criterion
        self.use_mlp = use_mlp
        self.use_fps_branch = use_fps_branch
        self.use_tps = use_tps
        self.use_nocs_refiner = use_nocs_refiner
        self.disable_pc_nocs_refine_in_test = disable_pc_nocs_refine_in_test
        self.disable_mesh_nocs_refine_in_test = disable_mesh_nocs_refine_in_test
        self.mesh_loss_weight = mesh_loss_weight

        self.surface_criterion = criterion
        if self.transformer_siamese.nocs_bins is not None:
            self.nocs_criterion = nn.CrossEntropyLoss()
        else:
            self.nocs_criterion = criterion
        self.mesh_criterion = criterion

        self.volume_loss_weight = volume_loss_weight
        self.nocs_loss_weight = nocs_loss_weight
        self.warp_loss_weight = warp_loss_weight
        self.fps_branch_loss_weight = fps_branch_loss_weight

        self.learning_rate = learning_rate
        assert optimizer_type in ('Adam', 'SGD')
        self.optimizer_type = optimizer_type
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        self.debug = debug

    # forward function for each stage
    # ===============================
    def encoder_forward(self, data, is_train=False):
        input1 = ME.SparseTensor(data['feat1'], coordinates=data['coords1'])
        input2 = ME.SparseTensor(data['feat2'], coordinates=data['coords2'])

        features_frame1_sparse, _ = self.sparse_unet3d_encoder(input1)
        features_frame2_sparse, _ = self.sparse_unet3d_encoder(input2)
        features_frame1 = features_frame1_sparse.F
        features_frame2 = features_frame2_sparse.F

        pc_nocs_frame1 = data['y1']
        pos_gt_frame2 = data['y2']
        per_point_batch_idx_frame1 = data['pc_batch_idx1']
        per_point_batch_idx_frame2 = data['pc_batch_idx2']
        sim_points_frame1 = data['pos1']
        sim_points_frame2 = data['pos2']

        if (self.transformer_siamese.encoder_pos_embed_input_dim == 6 and    # 训练时为真
            not self.transformer_siamese.inverse_source_template)\
                or (self.transformer_siamese.decoder_pos_embed_input_dim == 6 and   # 3
                    self.transformer_siamese.inverse_source_template):              # False
            frame1_coord = torch.cat([sim_points_frame1, pc_nocs_frame1], dim=-1)
        elif self.transformer_siamese.encoder_pos_embed_input_dim == 3:
            frame1_coord = sim_points_frame1
        else:
            raise NotImplementedError

        fusion_feature, pc_logits_frame2 = self.transformer_siamese(features_frame1, frame1_coord,
                                                                    features_frame2, sim_points_frame2)

        if self.transformer_siamese.nocs_bins is not None: # 64
            # NOCS classification
            vg = self.transformer_siamese.get_virtual_grid(pc_logits_frame2.get_device())
            nocs_bins = self.transformer_siamese.nocs_bins
            pred_logits_bins = pc_logits_frame2.reshape(
                (pc_logits_frame2.shape[0], nocs_bins, 3))
            nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
            pc_nocs_frame2 = vg.idxs_to_points(nocs_bin_idx_pred)
        else:
            # NOCS regression
            pc_nocs_frame2 = pc_logits_frame2

        nocs_data = dict(
            x_frame2=fusion_feature,
            pos_frame1=pc_nocs_frame1,
            pos_frame2=pc_nocs_frame2,
            logits_frame2=pc_logits_frame2,
            pos_gt_frame2=pos_gt_frame2,
            batch_frame1=per_point_batch_idx_frame1,
            batch_frame2=per_point_batch_idx_frame2,
            sim_points_frame1=sim_points_frame1,
            sim_points_frame2=sim_points_frame2)

        if self.use_nocs_refiner:
            assert self.transformer_siamese.nocs_bins is not None
            mesh_nocs = data['surf_query_points2']
            batch_size = torch.max(per_point_batch_idx_frame2).item() + 1
            gt_mesh_nocs = data['gt_surf_query_points2'] if is_train else None
            refined_pc_logits_frame2, refined_surf_query_points2 = \
                self.nocs_refiner(pc_nocs_frame2, sim_points_frame2, pc_logits_frame2, fusion_feature, mesh_nocs,
                                  batch_size, is_train=is_train, gt_mesh_nocs=gt_mesh_nocs)
            if self.disable_mesh_nocs_refine_in_test:
                nocs_data['refined_surf_query_points2'] = mesh_nocs
            else:
                nocs_data['refined_surf_query_points2'] = refined_surf_query_points2
            if self.disable_pc_nocs_refine_in_test:
                refined_pc_logits_frame2 = pc_logits_frame2

            nocs_data['refined_logits_frame2'] = refined_pc_logits_frame2

            # get NOCS coordinates from logits
            vg = self.transformer_siamese.get_virtual_grid(pc_logits_frame2.get_device())
            nocs_bins = self.transformer_siamese.nocs_bins
            refined_pred_logits_bins = refined_pc_logits_frame2.reshape(
                (refined_pc_logits_frame2.shape[0], nocs_bins, 3))
            refined_nocs_bin_idx_pred = torch.argmax(refined_pred_logits_bins, dim=1)
            refined_pc_nocs_frame2 = vg.idxs_to_points(refined_nocs_bin_idx_pred)
            nocs_data['refined_pos_frame2'] = refined_pc_nocs_frame2

        return nocs_data
    
    def unet3d_forward(self, encoder_result, is_train=False):
        # volume agg
        in_feature_volume = self.volume_agg(encoder_result, self.batch_size, is_train)

        # unet3d
        out_feature_volume = self.unet_3d(in_feature_volume)
        unet3d_result = {
            'out_feature_volume': out_feature_volume
        }
        return unet3d_result
    
    def fps_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.fps_like_surface_decoder(out_feature_volume, query_points)
        fps_output = self.last_fps_decoder({"out_features": out_features})
        decoder_result = {
            'out_feature_fps': fps_output
        }
        return decoder_result

    def surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result

    def tps_forward_v1(self, surface_decoder_result, unet3d_result, encoder_result, batchsize):
        surface_decoder_point = surface_decoder_result['out_features']
        tps_output = self.tps_module(surface_decoder_point, unet3d_result, encoder_result, batchsize)
        return tps_output

    # forward
    # =======
    def forward(self, data, is_train=False, use_refine_mesh_for_query=True):
        encoder_result = self.encoder_forward(data, is_train)
        if is_train:
            surface_query_points = data['gt_surf_query_points2']
        else:
            if use_refine_mesh_for_query:
                surface_query_points = encoder_result['refined_surf_query_points2']
            else:
                surface_query_points = data['surf_query_points2']
        unet3d_result = self.unet3d_forward(encoder_result, is_train)
        # if self.use_tps:
        #     batch_size = data['dataset_idx1'].size()[0]
        #     encoder_result['gt_surf_query_points2'] = data['gt_surf_query_points2']
        #     tps_result = self.tps_forward_v1(unet3d_result, encoder_result, batch_size, is_train)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)
        if self.use_tps:
            batch_size = data['dataset_idx1'].size()[0]
            encoder_result['surf_query_points2'] = surface_query_points
            tps_result = self.tps_forward_v1(surface_decoder_result, unet3d_result, encoder_result, batch_size)

        result = {
            'encoder_result': encoder_result,
            'unet3d_result': unet3d_result,
            'surface_decoder_result': surface_decoder_result
        }
        if self.use_fps_branch:
            fps_decoder_result = self.fps_decoder_forward(
                unet3d_result, surface_query_points)
            result['fps_decoder_result'] = fps_decoder_result
        if self.use_tps:
            result['tps_result'] = tps_result
        # surface_docoder_fps_result = self.fps_forward(surface_decoder_result)
        # surface_docoder_fps_result = self.fps_forward_learnable_linear(surface_decoder_result)
        # result = {
        #     'encoder_result': encoder_result,
        #     'unet3d_result': unet3d_result,
        #     'surface_decoder_result': surface_decoder_result
        # }
        return result

    # training
    # ========
    def configure_optimizers(self):
        if self.optimizer_type == 'Adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return NotImplementedError

    def vis_batch(self, batch, batch_idx, result, is_train=False, img_size=256):
        nocs_data = result['encoder_result']
        pred_nocs = nocs_data['pos_frame2'].detach()
        if self.use_nocs_refiner:
            refined_pred_pc_nocs = nocs_data['refined_pos_frame2'].detach()
            refined_pred_mesh_nocs = nocs_data['refined_surf_query_points2'].detach()
        gt_nocs = nocs_data['pos_gt_frame2']
        batch_idxs = nocs_data['batch_frame2']
        this_batch_size = len(batch['dataset_idx1'])
        gt_mesh_nocs = batch['gt_surf_query_points2']

        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        if is_train:
            max_vis_per_epoch = self.max_vis_per_epoch_train
            prefix = 'train_'
        else:
            max_vis_per_epoch = self.max_vis_per_epoch_val
            prefix = 'val_'

        _, selected_idxs, vis_idxs = get_vis_idxs(batch_idx, 
            batch_size=batch_size, this_batch_size=this_batch_size, 
            vis_per_items=vis_per_items, max_vis_per_epoch=max_vis_per_epoch)
        
        log_data = dict()
        for i, vis_idx in zip(selected_idxs, vis_idxs):
            label = prefix + str(vis_idx)
            is_this_item = (batch_idxs == i)
            this_gt_nocs = to_numpy(gt_nocs[is_this_item])
            this_pred_nocs = to_numpy(pred_nocs[is_this_item])
            if self.use_nocs_refiner:
                refined_label = prefix + 'refine_pc_' + str(vis_idx)
                this_refined_pred_pc_nocs = to_numpy(refined_pred_pc_nocs[is_this_item])
                refined_pc_nocs_img = render_nocs_pair(this_gt_nocs, this_refined_pred_pc_nocs,
                                                    None, None, img_size=img_size)
                log_data[refined_label] = [wandb.Image(refined_pc_nocs_img, caption=refined_label)]

                refined_label = prefix + 'refine_mesh_' + str(vis_idx)
                this_refined_pred_mesh_nocs = to_numpy(refined_pred_mesh_nocs.reshape(this_batch_size, -1, 3)[i])
                this_gt_mesh_nocs = to_numpy(gt_mesh_nocs.reshape(this_batch_size, -1, 3)[i])
                refined_mesh_nocs_img = render_nocs_pair(this_gt_mesh_nocs, this_refined_pred_mesh_nocs,
                                                    None, None, img_size=img_size)
                log_data[refined_label] = [wandb.Image(refined_mesh_nocs_img, caption=refined_label)]

            nocs_img = render_nocs_pair(this_gt_nocs, this_pred_nocs, 
                None, None, img_size=img_size)
            img = nocs_img
            log_data[label] = [wandb.Image(img, caption=label)]

        return log_data

    def infer(self, batch, batch_idx, is_train=True):
        if len(batch) == 0:
            return dict(loss=torch.tensor(0., device='cuda:0', requires_grad=True))
        try:
            result = self(batch, is_train=is_train)
            encoder_result = result['encoder_result']

            # NOCS error distance
            pred_nocs = encoder_result['pos_frame2']
            gt_nocs = encoder_result['pos_gt_frame2']
            nocs_err_dist = torch.norm(pred_nocs - gt_nocs, dim=-1).mean()

            # NOCS loss
            if self.transformer_siamese.nocs_bins is not None:
                # classification
                nocs_bins = self.transformer_siamese.nocs_bins
                vg = self.transformer_siamese.get_virtual_grid(pred_nocs.get_device())
                pred_logits = encoder_result['logits_frame2']
                pred_logits_bins = pred_logits.reshape(
                    (pred_logits.shape[0], nocs_bins, 3))
                gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
                nocs_loss = self.nocs_criterion(pred_logits_bins, gt_nocs_idx) * self.nocs_loss_weight
            else:
                # regression
                nocs_loss = self.nocs_criterion(pred_nocs, gt_nocs) * self.nocs_loss_weight

            if self.use_nocs_refiner:
                # only support classification
                assert self.transformer_siamese.nocs_bins is not None
                nocs_bins = self.transformer_siamese.nocs_bins
                refined_pred_nocs = encoder_result['refined_pos_frame2']
                refined_nocs_err_dist = torch.norm(refined_pred_nocs - gt_nocs, dim=-1).mean()

                vg = self.transformer_siamese.get_virtual_grid(refined_pred_nocs.get_device())
                refined_pred_logits = encoder_result['refined_logits_frame2']
                refined_pred_logits_bins = refined_pred_logits.reshape(
                    (refined_pred_logits.shape[0], nocs_bins, 3))
                gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
                refined_nocs_loss = self.nocs_criterion(refined_pred_logits_bins, gt_nocs_idx) * self.nocs_loss_weight

                refined_mesh_nocs = encoder_result['refined_surf_query_points2']
                gt_mesh_nocs = batch['gt_surf_query_points2']
                mesh_loss = self.mesh_loss_weight * self.mesh_criterion(refined_mesh_nocs, gt_mesh_nocs)
                refined_mesh_err_dist = torch.norm(refined_mesh_nocs - gt_mesh_nocs, dim=-1).mean()

            # warp field loss (surface loss)
            surface_decoder_result = result['surface_decoder_result']
            surface_criterion = self.surface_criterion
            pred_warp_field = surface_decoder_result['out_features']
            gt_sim_points_frame2 = batch['gt_sim_points2']
            gt_warpfield = gt_sim_points_frame2.reshape(pred_warp_field.shape)
            if self.use_tps:
                tps_result = result['tps_result']
                theta_tps, grid_tps, point_tps = tps_result['theta_tps'], tps_result['grid_tps'], tps_result['point_tps']
                theta_tps_loss = self.tps_theta_criterion(theta_tps) * self.tps_theta_loss_weight
                grid_tps_loss = self.tps_grid_criterion(grid_tps) * self.tps_grid_loss_weight
                point_tps_loss = self.tps_point_criterion(point_tps, gt_warpfield[..., [0, 2]])
            if self.use_fps_branch:
                pred_fps_points = result['fps_decoder_result']['out_feature_fps']['out_features']
                with torch.no_grad():
                    gt_fps_branch = self.last_fps_decoder({'out_features': gt_sim_points_frame2.view(pred_warp_field.shape[0], -1, 3)})['out_features']
                fps_branch_loss = self.fps_branch_loss_weight * self.surface_criterion(pred_fps_points, gt_fps_branch)

            if self.use_tps:
                warp_loss = surface_criterion(pred_warp_field[..., [1]], gt_warpfield[..., [1]])
            else:
                warp_loss = surface_criterion(pred_warp_field, gt_warpfield)
            warp_loss = self.warp_loss_weight * warp_loss

            loss_dict = {
                'nocs_err_dist': nocs_err_dist,
                'nocs_loss': nocs_loss,
                'warp_loss': warp_loss,
            }
            if self.use_nocs_refiner:
                loss_dict['refined_nocs_loss'] = refined_nocs_loss
                loss_dict['refined_nocs_err_dist'] = refined_nocs_err_dist
                loss_dict['mesh_loss'] = mesh_loss
                loss_dict['refined_mesh_err_dist'] = refined_mesh_err_dist
            if self.use_fps_branch:
                loss_dict['fps_branch_loss'] = fps_branch_loss
            if self.use_tps:
                loss_dict['theta_tps_loss'] = theta_tps_loss
                loss_dict['grid_tps_loss'] = grid_tps_loss
                loss_dict['point_tps_loss'] = point_tps_loss
            for key, value in loss_dict.items():
                print(key, value)
            metrics = dict(loss_dict)
            metrics['loss'] = nocs_loss + warp_loss
            if self.use_nocs_refiner:
                metrics['loss'] += refined_nocs_loss
                metrics['loss'] += mesh_loss
            if self.use_fps_branch:
                metrics['loss'] += fps_branch_loss
            if self.use_tps:
                metrics['loss'] += theta_tps_loss
                metrics['loss'] += grid_tps_loss
                metrics['loss'] += point_tps_loss

            for key, value in metrics.items():
                log_key = ('train_' if is_train else 'val_') + key
                self.log(log_key, value)
            log_data = self.vis_batch(batch, batch_idx, result, is_train=is_train)
            self.logger.log_metrics(log_data, step=self.global_step)
        except Exception as e:
            raise e

        return metrics

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']
