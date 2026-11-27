# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import numpy as np
from copy import copy

import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS
from ...datasets.pipelines.loading import LoadAnnotationsBEVDepth
from mmdet.core import multi_apply
from .bevdet import BEVDepth4D
import clip
import torch.nn as nn
import math



class CrossAttention(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_q = nn.Linear(input_dim, output_dim)
        self.fc_k = nn.Linear(input_dim, output_dim)
        self.fc_v = nn.Linear(input_dim, output_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = self.softmax(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights

class ImageVpnGenerator(torch.nn.Module):
    def __init__(self, clip_dim, reduction=1, dist_type='gaussian'):
        super(ImageVpnGenerator, self).__init__()
        self.clip_dim = clip_dim
        self.reduction = reduction
        self.hidden_dim = self.clip_dim // self.reduction
        self.dist_type = dist_type
        # self.grad_clip_val = nn.Parameter(torch.tensor(3.0)) 
        self._build_up()

    def _build_up(self):

        self.crossAttention = CrossAttention(self.clip_dim, self.hidden_dim)
        # 修改方差层的初始化
        self.fc_variance = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.clip_dim),
            # 将BatchNorm1d改为LayerNorm以适应序列数据
            torch.nn.LayerNorm(self.clip_dim),
            torch.nn.ReLU()  # 保持原有结构
        )
        self.fc_mean = torch.nn.Linear(self.hidden_dim, self.clip_dim)
        nn.init.xavier_uniform_(self.fc_variance[0].weight, gain=0.1)
        nn.init.constant_(self.fc_variance[0].bias, 0.01)
        nn.init.xavier_uniform_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0.1)

    def forward(self, spatial_feat, text_feat):
        
        text_feat = text_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
        text_feat = text_feat.expand(-1, spatial_feat.size(1), -1)  # [B, H*W, C]
        
        # 交叉注意力计算
        attn_feat, _ = self.crossAttention(spatial_feat, text_feat, text_feat)
        
        # 生成分布参数
        variance = self.fc_variance(attn_feat)
        mu = self.fc_mean(attn_feat)
        mu = torch.sigmoid(mu) * 0.8 + 0.1  # 限制在[0.1, 0.9]
        variance = torch.sigmoid(variance)   # 限制方差在[0,1]
        return mu, variance  # 输出形状 [B, H*W, C]

    def sample(self, mu, variance):
        if self.dist_type == 'gaussian':
            noise = torch.randn_like(mu) * variance + mu
        elif self.dist_type == 'beta':
            # 添加温度系数控制分布范围
            temperature = 0.1
            # 使用softplus确保参数为正
            alpha = F.softplus(mu * variance / temperature) + 1e-6
            beta = F.softplus((1 - mu) * variance / temperature) + 1e-6
            # 添加参数归一化
            sum_ab = alpha + beta
            alpha = alpha / sum_ab * 5.0 + 1e-6  # 控制参数范围在[1e-6,5]
            beta = beta / sum_ab * 5.0 + 1e-6
            noise = torch.distributions.Beta(alpha, beta).rsample()
        elif self.dist_type == 'uniform':
            # 将方差解释为分布范围
            a = mu - variance.abs() 
            b = mu + variance.abs()
            noise = torch.rand_like(mu) * (b - a) + a
        else:
            raise ValueError(f"Unsupported distribution: {self.dist_type}")
            
        return noise

class MoCAAWithSoftPrompt(nn.Module):
    def __init__(self, in_dim, prompt_channels=64, height=180, width=180):
        super(MoCAAWithSoftPrompt, self).__init__()
        self.in_dim = in_dim
        self.prompt_channels = prompt_channels
        self.height = height
        self.width = width

        # Soft prompt 是 learnable 的张量
        self.soft_prompt = nn.Parameter(
            torch.randn(1, prompt_channels, height, width) * 0.02
        )

        # prompt 拼接后 -> 降维回 in_dim
        self.reduce_conv = nn.Conv2d(in_dim + prompt_channels, in_dim, kernel_size=1)

        # 原始 MoCAA 模块
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.conv_out = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == (self.height, self.width), f"Expected spatial size {(self.height, self.width)}, but got {(H, W)}"

        # 复制 soft prompt 到每个 batch
        prompt = self.soft_prompt.expand(B, -1, -1, -1)  # [B, Cp, H, W]

        # 拼接到输入特征图上
        x = torch.cat([x, prompt], dim=1)  # [B, C + Cp, H, W]
        # import ipdb;ipdb.set_trace()
        x = self.reduce_conv(x)            # 降回 [B, C, H, W]

        # ----- 以下是原始 MoCAA 操作 -----
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query_H = query.permute(0, 3, 1, 2).contiguous().view(B, W, -1)
        key_H = key.permute(0, 3, 2, 1).contiguous().view(B, W, -1)
        # import ipdb;ipdb.set_trace()
        attention_H = torch.bmm(query_H, key_H.transpose(1, 2))
        attention_H = F.softmax(attention_H, dim=-1)
        value_H = value.permute(0, 3, 2, 1).contiguous().view(B, W, -1)
        out_H = torch.bmm(attention_H, value_H).view(B, W, C, H).permute(0, 2, 3, 1)

        query_W = query.permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        key_W = key.permute(0, 2, 3, 1).contiguous().view(B, H, -1)
        attention_W = torch.bmm(query_W, key_W.transpose(1, 2))
        attention_W = F.softmax(attention_W, dim=-1)
        value_W = value.permute(0, 2, 3, 1).contiguous().view(B, H, -1)
        out_W = torch.bmm(attention_W, value_W).view(B, H, C, W).permute(0, 2, 1, 3)

        out = self.gamma * (out_H + out_W) + x
        out = self.conv_out(out)
        out = self.norm(out)
        return out
    
@DETECTORS.register_module()
class CLIPBEV(BEVDepth4D):

    def __init__(self, scale_num=3, bev_paste=True, prev_num=1, img_backbone=None,
                 bda_aug_conf=None, text_encoder=None, score_map_idx = [2], #  prompt_generator=None, 
                 vis_context_decoder=None,
                 text_context_decoder=None,
                 context_length = 25,
                 in_dim = 80,
                 fg_norm_shape = (16, 44),
                 **kwargs):
        super(CLIPBEV, self).__init__(**kwargs)
        self.scale_num = scale_num
        self.bev_paste = bev_paste
        self.prev_num = prev_num
        self.prev_data = [dict() for i in range(prev_num)]
        self.bda_aug_conf = bda_aug_conf
        self.text_encoder = DETECTORS.build(text_encoder)
        # self.prompt_generator = DETECTORS.build(prompt_generator)
        self.vis_context_decoder = DETECTORS.build(vis_context_decoder)
        self.text_context_decoder = DETECTORS.build(text_context_decoder)
        self.img_backbone = DETECTORS.build(img_backbone)
        self.context_length = context_length
        self.tau = nn.Parameter(torch.tensor(0.5))  # 初始值从0.07提升到0.5
        self.score_map_idx = score_map_idx
        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian','traffic_cone'
        ]
        self.num = len(self.class_names)
        if context_length > 10:
            self.texts = torch.cat(
                [clip.tokenize(f"In this photo, a {label} can be found.",
                               context_length=self.context_length) for label in self.class_names])
        else:
            self.texts = torch.cat(
                [clip.tokenize(label, context_length=self.context_length) for label in self.class_names])

        # learnable textual context
        learnable_context = self.text_encoder.context_length - self.context_length
        token_embed_dim = self.text_encoder.transformer_width
        text_dim = self.text_encoder.embed_dim

        self.contexts = nn.Parameter(torch.randn(1, learnable_context, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.vis_gamma = nn.Parameter(torch.ones([]) * 1e-3)  # for updating visual embedding diff
        # self.imageVpnGenerator = ImageVpnGenerator(self.text_encoder.embed_dim, dist_type='gaussian')
        self.mocab = MoCAAWithSoftPrompt(in_dim=in_dim, prompt_channels=64, height=256, width=256)
        # self.cdfa = ContrastDrivenFeatureAggregation(80, 80, 4)
        # self.preprocess_fg = UNetPreprocess(80, 80, 8)  # 1/2
        # self.preprocess_bg = UNetPreprocess(80, 80, 8)  # 1/2
        # 新增MLP层
        self.fg_aggregator = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.GroupNorm(8, 64),  # 替换BatchNorm为GroupNorm
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),  # 更细粒度的分组
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.LayerNorm([1, *fg_norm_shape]),
            nn.Hardtanh(min_val=-5, max_val=5)  # 缩小输出范围
        )
        # 改进初始化策略
        for m in self.fg_aggregator:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')  # 适配LeakyReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 最后一层特殊初始化
        nn.init.constant_(self.fg_aggregator[-3].weight, 0.01)
        nn.init.constant_(self.fg_aggregator[-3].bias, -2.0)
        
        if self.bev_paste:
            if bda_aug_conf is None:
                bda_aug_conf = dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5)
            self.loader = LoadAnnotationsBEVDepth(bda_aug_conf, None, is_train=True)
            
    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, prev_data, bda_paste, frame_id, is_train=False, img_metas=None):
        x, score_maps = self.image_encoder(img)
        bev_feat, img_preds = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, prev_data, bda_paste, img], frame_id, is_train=is_train)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, img_preds, score_maps
    
    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)

        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra

        if len(inputs) == 9:
            prev_data = inputs[7]
            bda_paste = inputs[8]
        else:
            prev_data = None
            bda_paste = None
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor, prev_data, bda_paste

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = list(self.img_backbone(imgs))
        # import ipdb;ipdb.set_trace()
        
        score_maps, text_features = self.compute_text_features(x)
        
        # import ipdb;ipdb.set_trace()
        x = list(x[:-1])
        # 多尺度特征融合（将文本注意力图拼接到指定层级）
        for i in self.score_map_idx:
            if x[i].shape[2:] != score_maps.shape[-2:]:
                resized_mask = F.interpolate(score_maps.contiguous(), size=x[i].shape[2:], mode="bilinear", align_corners=True)
                x[i] = torch.cat([x[i], resized_mask], dim=1)
            else:
                x[i] = torch.cat([x[i], score_maps.contiguous()], dim=1)
        if self.with_img_neck:
            x[self.scale_num - 1] = self.img_neck(x[self.scale_num - 1:])
            if type(x) in [list, tuple]:
                x = x[:self.scale_num]
        # import ipdb;ipdb.set_trace()
        for i in range(self.scale_num):
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        return x[:self.scale_num][::-1], score_maps    
    def compute_text_features(self, x):
        """计算文本特征与视觉特征的交互"""
        # 从主干网络第5层特征中获取全局特征和视觉嵌入
        # global_feat: [B, 1024] 全局图像特征
        # visual_embeddings: [B, 1024, 32, 32] 空间视觉特征
        # import ipdb;ipdb.set_trace()
        global_feat, visual_embeddings = x[4] 

        # 构建视觉上下文：全局特征 + 空间特征展平
        B, C, H, W = visual_embeddings.shape
        visual_context = torch.cat([
            global_feat.reshape(B, C, 1),  # 全局特征作为上下文锚点
            visual_embeddings.reshape(B, C, H*W)  # 空间特征展开为序列
        ], dim=2).permute(0, 2, 1)  # 转换为 [B, N, C] 格式

        # 生成文本提示（当前被注释，暂时用None代替）
        prompt_gen = None

        # 文本编码器生成文本特征
        # self.texts: 预定义的类别文本描述
        # self.contexts: 可学习的文本上下文参数
        text_embeddings = self.text_encoder(
            self.texts.to(global_feat.device), 
            self.contexts, 
            prompt_gen
        ).expand(B, -1, -1).contiguous()  # 扩展为 [B, 15, 1024]

        # 视觉-文本交互解码
        vis_prompt_diff = self.vis_context_decoder(
            visual_embeddings.reshape(B, C, H*W).permute(0, 2, 1).contiguous(),
            text_embeddings
        )  # 视觉特征在文本指导下的调整量
        
        # 更新视觉特征（带学习率系数）

        visual_embeddings = visual_embeddings + self.vis_gamma * vis_prompt_diff.permute(0, 2, 1).view(B, C, H, W).contiguous()

        # 文本特征更新（基于视觉上下文）
        text_diff = self.text_context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff #[4, 15, 1024]

        # 特征归一化与相似度计算
        text_embeddings_loss = text_embeddings
        visual_embeddings_loss = visual_embeddings
        text_features = F.normalize(text_embeddings, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        # import pdb;pdb.set_trace()
        # 生成最终得分图：[B, 15, H, W]
        # 在特征归一化后添加以下代码
        # import ipdb;ipdb.set_trace()
        visual_embeddings_flat = visual_embeddings.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        # mu, variance = self.imageVpnGenerator(visual_embeddings_flat, text_embeddings)
        # visual_embeddings = self.imageVpnGenerator.sample(mu, variance)  # 采样生成增强特征
        visual_embeddings = visual_embeddings.view(B, C, H, W)  # 恢复原始形状
        score_map_attnpool = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        return score_map_attnpool, text_embeddings_loss
    
    def extract_img_feat(self, img, img_metas, is_train=False, **kwargs):
        imgs, rots, trans, intrins, post_rots, post_trans, bda, _, prev_data, bda_paste = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        img_preds_list = []
        key_frame = True  # back propagation for key frame only
        for frame_id, (img, rot, tran, intrin, post_rot, post_tran) in enumerate(zip(
                imgs, rots, trans, intrins, post_rots, post_trans)):
            if key_frame or self.with_prev:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, prev_data, bda_paste)
                # print(prev_data)
                if key_frame:
                    bev_feat, img_preds, score_maps = self.prepare_bev_feat(*inputs_curr, frame_id, is_train, img_metas)
                else:
                    with torch.no_grad():
                        bev_feat, img_preds, score_maps = self.prepare_bev_feat(*inputs_curr, frame_id, is_train)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                img_preds = None
            bev_feat_list.append(bev_feat)
            
            img_preds_list.append(img_preds)
            key_frame = False
        # import ipdb;ipdb.set_trace()
        # f_fg1 = self.preprocess_fg(bev_feat_list[0])
        # f_bg1 = self.preprocess_bg(bev_feat_list[0])
        # bev_feat_list[0] = self.cdfa(bev_feat_list[0], f_fg1, f_bg1)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        bev_feat = self.mocab(bev_feat)
        x = self.bev_encoder(bev_feat)

        # x = checkpoint(self.bev_encoder,bev_feat)
        
        return [x], img_preds_list, score_maps
    
    def extract_feat(self, points, img, img_metas, Train=False, **kwargs):
        """Extract features from images and points."""
        img_feats, img_preds_list, score_maps = self.extract_img_feat(img, img_metas, **kwargs, is_train=Train)
        pts_feats = None
        if Train:
            return (img_feats, pts_feats, img_preds_list, score_maps)
        else:
            return (img_feats, pts_feats, img_preds_list)
    def compute_fg_supervision(self, score_maps, gt_fg):
        """
        前景监督计算函数
        参数:
            score_maps: [B*N, C, H, W] 分类得分图 (B=8, N=6, C=10)
            gt_fg: [B, N, H_gt, W_gt] 原始前景标签 (H_gt=256, W_gt=704)
        返回:
            loss: 标量损失值
        """
        # 维度拆解
        B, N = gt_fg.shape[:2]
        num_classes = score_maps.shape[1]
        
        # 直接使用原始logits [B*N, C, H, W] -> [B, N, C, H, W]
        score_maps = score_maps.view(B, N, num_classes, *score_maps.shape[-2:])
        
        # 取所有类别的最大logit作为前景预测 [B, N, H, W]
        # 处理每个样本的类别维度
        B, N, C, H, W = score_maps.shape
        score_maps = score_maps.view(B*N, C, H, W)
        pred_fg_agg = self.fg_aggregator(score_maps)  # [B*N, 1, H, W]
        pred_fg_agg = pred_fg_agg.view(B, N, H, W)
        
        # 下采样gt_fg到score_maps的尺寸 [B, N, 256,704] -> [B, N, 8,22]
        gt_fg_down = F.adaptive_avg_pool2d(gt_fg.view(-1, *gt_fg.shape[-2:]), 
                                          output_size=score_maps.shape[-2:])
        gt_fg_down = gt_fg_down.view(B, N, *gt_fg_down.shape[-2:])
        
        # 使用BCEWithLogitsLoss替代sigmoid+BCE
        loss = F.binary_cross_entropy_with_logits(
            pred_fg_agg,          # [B, N, 8,22]
            gt_fg_down.float()    # [B, N, 8,22]
        )
        return loss
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        
        gt_bboxes_inbox = gt_bboxes_3d
        if self.bev_paste:
            gt_bboxes_cur = copy(gt_bboxes_3d)
            gt_labels_cur = copy(gt_labels_3d)
            B = len(gt_labels_3d)
            bda_paste = []
            gt_bboxes_paste = []
            gt_labels_paste = []
            gt_bboxes_inbox = []
            # adding the samples in previous step to the current samples for BEV-Paste
            # the batch is expanded by prev_num times after BEV-Paste
            for i in range(self.prev_num):
                bda_mat_paste = []
                if self.prev_data[i]:
                    for j in range(B):
                        gt_bboxes_tmp = torch.cat([gt_bboxes_3d[j].tensor.clone(), 
                            self.prev_data[i]['gt_bboxes'][j].tensor.clone()], dim=0)
                        gt_labels_tmp = torch.cat([gt_labels_3d[j].clone(), 
                            self.prev_data[i]['gt_labels'][j].clone()], dim=0)
                        
                        rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                        gt_bboxes_tmp, bda_rot = self.loader.bev_transform(gt_bboxes_tmp.cpu(), 
                                                                rotate_bda, scale_bda, flip_dx, flip_dy)
                        gt_bboxes_inbox_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp[:len(gt_labels_3d[j])])
                        gt_bboxes_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp.cuda())

                        bda_mat_paste.append(bda_rot.cuda())
                        gt_bboxes_paste.append(gt_bboxes_tmp)
                        gt_labels_paste.append(gt_labels_tmp)
                        if i==0:
                            gt_bboxes_inbox.append(gt_bboxes_inbox_tmp)
                else:
                    for j in range(B):
                        gt_bboxes_tmp = gt_bboxes_3d[j].tensor.clone()
                        gt_labels_tmp = gt_labels_3d[j].clone()
                        
                        rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                        gt_bboxes_tmp, bda_rot = self.loader.bev_transform(gt_bboxes_tmp.cpu(), 
                                                                rotate_bda, scale_bda, flip_dx, flip_dy)
                        gt_bboxes_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp.cuda())

                        bda_mat_paste.append(bda_rot.cuda())
                        gt_bboxes_paste.append(gt_bboxes_tmp)
                        gt_labels_paste.append(gt_labels_tmp)
                        if i==0:
                            gt_bboxes_inbox.append(gt_bboxes_tmp)
                bda_paste.append(torch.stack(bda_mat_paste))

            img_inputs.append(self.prev_data)
            img_inputs.append(bda_paste)
            gt_bboxes_3d = gt_bboxes_paste
            gt_labels_3d = gt_labels_paste


        img_feats, pts_feats, img_preds_list, score_maps = self.extract_feat(
            points, Train=True, img=img_inputs, img_metas=img_metas, **kwargs)
        # import ipdb;ipdb.set_trace()
        pred_depth = img_preds_list[0]['depths']
        pred_fg = img_preds_list[0]['fgs']
        coor = img_preds_list[0]['coors']
        gt_fg = kwargs['gt_fg']
        
        # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        gt_depth = kwargs['gt_depth']
        loss_fg, loss_inbox = \
            self.img_view_transformer.get_losses(gt_fg, gt_depth, pred_fg, pred_depth, 
                                                 gt_bboxes_inbox, coor)
        losses = dict(loss_fg=loss_fg, loss_inbox=loss_inbox)
        losses['loss_fg_supervise'] = self.compute_fg_supervision(score_maps, kwargs['gt_fg'])
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        if self.bev_paste:
            frustum_feats = []
            transforms = []
            for i in range(len(img_preds_list)):
                if img_preds_list[i] is not None:
                    frustum_feats.append(img_preds_list[i]['frustum_feat'].detach())
                    transforms.append(img_preds_list[i]['transform'])
                else:
                    frustum_feats.append(None)
                    transforms.append(None)
            cur_data = {'gt_bboxes': gt_bboxes_cur, 'gt_labels': gt_labels_cur, 
                        'frustum_feats':frustum_feats, 'transforms': transforms}
            self.prev_data.pop()
            self.prev_data.insert(0, cur_data)
        del img_preds_list
        return losses
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):

        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if num_augs==1:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas, img_inputs, **kwargs)
    
    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        img_feats, pts_feats, _= multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if img_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(img_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
