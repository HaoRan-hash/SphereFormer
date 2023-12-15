import torch
from torch import nn
import torch.nn.functional as F
import math


class RefineModel(nn.Module):
    def __init__(self, based_argmax=False, sigmoid_threshold=0.8, argmax_threshold=0.9, num_class=19, embed_dim=32):
        super().__init__()
        
        self.based_argmax = based_argmax
        self.sigmoid_threshold = sigmoid_threshold
        self.argmax_threshold = argmax_threshold
        self.num_class = num_class
        self.embed_dim = embed_dim
        if not based_argmax:
            self.select_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
                                            nn.BatchNorm1d(embed_dim, eps=1e-4),
                                            nn.ReLU(True),
                                            nn.Linear(embed_dim, 1, bias=True),
                                            nn.Sigmoid())
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.t_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        self.output_layer = nn.Linear(embed_dim, num_class)
    
    def forward(self, cvae_model, feat, scene_flow, coarse_pred, z_dim=16, target=None, ignore_label=None):
        """
        feat.shape = (n, c)
        scene_flow.shape = (n, 3)
        coarse_pred.shape = (n, num_class)  not softmax
        """
        device = feat.device
        coarse_pred = coarse_pred.softmax(dim=-1)
        
        # 只选择scene flow不为0的点进行增强
        temp_mask = ((scene_flow[:, 0] != 0) | (scene_flow[:, 1] != 0) | (scene_flow[:, 2] != 0))
        
        # select low quality feat
        if self.based_argmax:
            max_logit = coarse_pred.max(dim=-1)[0]
            valid_mask = (max_logit < self.argmax_threshold)
        else:
            quality_score = self.select_mlp(feat).squeeze(dim=-1)
            valid_mask = (quality_score < self.sigmoid_threshold)
            
            # debug
            # print(f'{(quality_score <= 0.5).float().sum()}, {(quality_score <= 0.6).float().sum()}, {(quality_score <= 0.7).float().sum()}, {(quality_score <= 0.8).float().sum()}, {(quality_score <= 0.9).float().sum()}, {(quality_score <= 1).float().sum()}')
            
            # 训练时计算bce loss
            if target is not None:
                score_target = (target == coarse_pred.argmax(dim=-1)).float()  # (n, )
                score_weight = torch.ones_like(target, device=device)
                score_weight[target == ignore_label] = 0
                bce_loss = F.binary_cross_entropy(quality_score, score_target, score_weight)
        
        valid_mask = valid_mask & temp_mask
        low_feat, low_scene_flow, low_coarse_pred = feat[valid_mask], scene_flow[valid_mask], coarse_pred[valid_mask]
        
        # debug
        # print(f'rank {torch.distributed.get_rank()}, {valid_mask.float().sum()}, {feat.shape}')
        
        # cvae generate feat
        c = torch.arange(0, self.num_class, step=1, device=device).unsqueeze(dim=0).expand(low_feat.shape[0], -1)   # (m, num_class)
        low_scene_flow = low_scene_flow.unsqueeze(dim=1).expand(-1, self.num_class, -1)   # (m, num_class, 3)
        z = torch.randn((1, 1, z_dim), device=device).expand(low_feat.shape[0], self.num_class, -1)   # (m, num_class, z_dim)
        c, low_scene_flow, z = c.flatten(0, 1), low_scene_flow.flatten(0, 1), z.flatten(0, 1)
        gen_feat = cvae_model(z, c, low_scene_flow)
        gen_feat = gen_feat.view(-1, self.num_class, self.embed_dim)   # (m, num_class, embed_dim)
        
        # augment low_feat based on attention
        gen_feat = gen_feat * low_coarse_pred.unsqueeze(dim=-1)
        
        q = self.q_proj(low_feat).unsqueeze(dim=1)
        k = self.k_proj(gen_feat)
        v = self.v_proj(gen_feat)
        
        attn_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_map = attn_map.softmax(dim=-1)   # (m, 1, num_class)
        
        aug_feat = torch.bmm(attn_map, v).squeeze(dim=1)
        aug_feat = self.o_proj(aug_feat)
        
        new_feat = torch.zeros_like(feat, device=device)
        new_feat[valid_mask] = aug_feat
        
        refine_feat = self.t_proj(torch.cat((feat, new_feat), dim=-1))
        
        # seg head
        output = self.output_layer(refine_feat)
        
        if target is not None:
            return output, bce_loss
        else:
            return output
