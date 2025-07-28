# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torch.nn.parallel.comm import broadcast

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from katransformer import KAN
# from kat_rational import KAT_Group
from rational_kat_cu.kat_rational import KAT_Group
from networks.uctnet_2D import DeepSupervision,uncertainty_map,UnoB,UnoB_fang,Vision_Transformer
from networks.vit_2D import Map_reshape
import torch.utils.checkpoint as checkpoint
from networks.UltraLight_VM_UNet import PVMLayer,PVMLayer_bing,PVMLayer_single
from networks.model_assa import TransformerBlock,assa,WindowAttention_sparse,WindowAttention_sparse_replace,window_partition
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Attention_replace(nn.Module):
    def __init__(self, config, vis):
        super(Attention_replace, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attn0 = self.softmax(attention_scores)
        attn1 = self.relu(attention_scores) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attention_probs = attn0 * w1 + attn1 * w2
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# class GRKAN(nn.Module):
#     def __init__(self, config):
#         super(GRKAN, self).__init__()
#         self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
#         self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = Dropout(config.transformer["dropout_rate"])
#
#         self._init_weights()
#
#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class GRKAN(nn.Module):
#     """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
#
#     def __init__(
#             self,
#             in_features,
#             hidden_features=None,
#             out_features=None,
#             act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
#             bias=True,
#             drop=0.,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#
#         self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
#         self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
#         self.drop1 = nn.Dropout(drop)
#         self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
#         self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
#         self.drop2 = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.act1(x)
#         x = self.drop1(x)
#         x = self.fc1(x)
#         x = self.act2(x)
#         x = self.drop2(x)
#         x = self.fc2(x)
#         return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # print(x.shape)
        # print(self.position_embeddings.shape)
        # print(x.shape)
        # print(self.position_embeddings.shape)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class KAM_Block(nn.Module):
    def __init__(self, config, vis):
        super(KAM_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # self.ffn = Mlp(config)
        self.ffn = KAN(
            in_features=config.hidden_size,
            hidden_features=int(config.hidden_size * 4),
            act_layer=nn.GELU,
            drop=0,
            act_init='gelu',
        )
        # self.attn = Attention(config, vis)
        # self.attn = Attention_replace(config, vis)
        # # print('config.transformer.mlp_dim:',config.transformer.mlp_dim)
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = Linear(config.hidden_size, self.all_head_size)
        # self.key = Linear(config.hidden_size, self.all_head_size)
        # self.value = Linear(config.hidden_size, self.all_head_size)

        self.win_size = 2
        self.attn = WindowAttention_sparse_replace(
            config.hidden_size, win_size=to_2tuple(self.win_size), num_heads=config.transformer["num_heads"],
            qkv_bias=True, qk_scale=None, attn_drop=config.transformer["attention_dropout_rate"], proj_drop=config.transformer["attention_dropout_rate"],
            token_projection='linear')
        self.mamba = PVMLayer_bing(config.hidden_size,config.hidden_size)
        # self.mamba = PVMLayer_single(config.hidden_size, config.hidden_size)
        # print(config.hidden_size)

    def forward(self, x):
        # print('x1:',x.shape)
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        h = x
        # x = self.attention_norm(x)
        # x = x.view(B, H, W, C)
        # x_windows = window_partition(x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        # x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # x, weights = self.attn(x_windows)
        # x = x.view(B, H * W, C)
        # x = x + h

        # print(1)
        x_mamba = self.mamba(x)
        x = x_mamba + h
        weights = 0

        # print('x2.shape:', x.shape)
        # x_mamba = self.mamba(h)
        # x = x + x_mamba

        # # B, L, C = x.shape
        # # H = int(math.sqrt(L))
        # # W = int(math.sqrt(L))
        # h = x
        # # part1, part2 = torch.chunk(x, 2,dim=2)
        # # x = self.attention_norm(part1)
        # # x = x.view(B, H, W, int(C/2))
        # # x_windows = window_partition(x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        # # x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # # x, weights = self.attn(x_windows)
        # # x = x.view(B, H * W, int(C/2))
        # # x = x + part1
        # x_mamba = self.mamba(x)
        # x = x_mamba + h
        # # x = torch.cat((x, x_mamba), dim=2)


        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            # query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            #
            # query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            # key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            # value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            # out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            # self.attn.query.weight.copy_(query_weight)
            # self.attn.key.weight.copy_(key_weight)
            # self.attn.value.weight.copy_(value_weight)
            # self.attn.out.weight.copy_(out_weight)
            # self.attn.query.bias.copy_(query_bias)
            # self.attn.key.bias.copy_(key_bias)
            # self.attn.value.bias.copy_(value_bias)
            # self.attn.out.bias.copy_(out_bias)

            # self.attn.qkv.to_q.query.weight.copy_(query_weight)
            # self.attn.key.weight.copy_(key_weight)
            # self.attn.value.weight.copy_(value_weight)
            # self.attn.out.weight.copy_(out_weight)
            # self.attn.query.bias.copy_(query_bias)
            # self.attn.key.bias.copy_(key_bias)
            # self.attn.value.bias.copy_(value_bias)
            # self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = KAM_Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            # dummy_input = torch.randn(hidden_states.shape).cuda()
            # flops, params = profile(layer_block, inputs=[dummy_input])
            # print(f"vit模型参数量: {params / 1e6:.2f}M")
            # print(f"vit模型FLOPs: {flops / 1e9:.2f}G")
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock_uct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dmodel,
            depths,
            input_resolution,
            num_heads,
            patch_size,
            dim_head,
            num_stage,
            bound_size,
            skip_channels=0,
            use_batchnorm=True,
            num_class = 2,

    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.num_pool = 4
        self.num_class = num_class
        self.deep_supervision = DeepSupervision(out_channels, self.num_class)
        self.deep_supervision_T = DeepSupervision(out_channels, self.num_class)
        self.vit_blocks = Vision_Transformer(dim=out_channels, dmodel=dmodel, input_resolution=input_resolution,
                                num_heads=num_heads, patch_size=patch_size, dropout=0.1,
                                in_depth=depths, add_Map=True, dim_head=dim_head)

        self.num_stage = num_stage
        self.bound_size = bound_size
        # self.mamba = PVMLayer(out_channels, out_channels)
        self.mamba = PVMLayer_bing(out_channels,out_channels)
        # self.assa = TransformerBlock(dim=out_channels, input_resolution=input_resolution, num_heads=num_heads,sparseAtt = True)
        self.assa = assa(dim=out_channels, input_resolution=input_resolution, num_heads=num_heads)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, skip=None, casename = '1'):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        ds = self.deep_supervision(x)

        if self.num_stage < self.num_pool:
            # UMT_Block
            s = x.detach()
            A_map = uncertainty_map(ds.clone().detach(), self.num_class, casename)
            A_map_pool = uncertainty_map(self.pool(ds.clone().detach()), self.num_class, casename)
            tar = torch.argmax(ds, dim=1, keepdim=True)
            tar_pool = torch.argmax(self.pool(ds), dim=1, keepdim=True)
            pro = UnoB_fang(tar[:, 0].long().detach().cpu().numpy(), A_map.cpu().numpy(), self.bound_size)
            pro_pool = UnoB_fang(tar_pool[:, 0].long().detach().cpu().numpy(), A_map_pool.cpu().numpy(), self.bound_size)
            A_map = torch.tensor(pro).cuda()
            A_map_pool = torch.tensor(pro_pool).cuda()
            s_shape = s.shape
            s = self.mamba(s.reshape(s_shape[0], s_shape[1], s_shape[2]*s_shape[3]).transpose(-1, -2), A_map.unsqueeze(1), A_map.unsqueeze(1))
            s_in = x.detach()
            A_map_in = A_map.view(A_map.shape[0], 1, A_map.shape[1], A_map.shape[2])
            A_map_in_pool = A_map_pool.view(A_map_pool.shape[0], 1, A_map_pool.shape[1], A_map_pool.shape[2])
            s_c = self.assa(s_in, mask=A_map_in, mask2 = A_map_in_pool)
            s = s.transpose(-1, -2).reshape(s_shape[0], s_shape[1], s_shape[2], s_shape[3]) + s_c
            s = self.vit_blocks(s, A_map, casename)
            if self.deep_supervision_T is not None:
                ds_T = self.deep_supervision_T(s)
            x = x + s

            return x, ds, A_map, ds_T
        else:
            return x, ds, None, None

# class DecoderBlock_uct(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             dmodel,
#             depths,
#             input_resolution,
#             num_heads,
#             patch_size,
#             dim_head,
#             num_stage,
#             bound_size,
#             skip_channels=0,
#             use_batchnorm=True,
#             num_class = 2,
#
#     ):
#         super().__init__()
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#
#         self.num_pool = 4
#         self.num_class = num_class
#         self.deep_supervision = DeepSupervision(out_channels, self.num_class)
#         self.deep_supervision_T = DeepSupervision(out_channels, self.num_class)
#         self.vit_blocks = Vision_Transformer(dim=out_channels, dmodel=dmodel, input_resolution=input_resolution,
#                                 num_heads=num_heads, patch_size=patch_size, dropout=0.1,
#                                 in_depth=depths, add_Map=True, dim_head=dim_head)
#
#         self.num_stage = num_stage
#         self.bound_size = bound_size
#         # self.mamba = PVMLayer(out_channels, out_channels)
#         self.mamba = PVMLayer_bing(out_channels,out_channels)
#         # self.assa = TransformerBlock(dim=out_channels, input_resolution=input_resolution, num_heads=num_heads,sparseAtt = True)
#         self.assa = assa(dim=out_channels, input_resolution=input_resolution, num_heads=num_heads)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x, skip=None, casename = '1'):
#         x = self.up(x)
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # print('x:',x.shape)
#
#         # if self.deep_supervision is not None:
#         # print('x:', torch.unique(x))
#         ds = self.deep_supervision(x)
#         # print('ds:', torch.unique(ds))
#         # print('ds:', ds.shape)
#         # print(torch.unique(ds))
#         if self.num_stage < self.num_pool:
#             # UMT_Block
#             s = x.detach()
#             A_map = uncertainty_map(ds.clone().detach(), self.num_class, casename)
#             A_map_pool = uncertainty_map(self.pool(ds.clone().detach()), self.num_class, casename)
#             tar = torch.argmax(ds, dim=1, keepdim=True)
#             tar_pool = torch.argmax(self.pool(ds), dim=1, keepdim=True)
#             # # 有broadcast
#             # # pro = UnoB(tar[:, 0].long().detach().cpu().numpy(), A_map.cpu().numpy(), self.bound_size)
#             # # A_map = torch.tensor(pro).cuda()
#             #
#             # # s = self.mamba(s, A_map.unsqueeze(1), A_map.unsqueeze(1))
#             # # s = checkpoint.checkpoint(self.vit_blocks, s, A_map)
#             # # s = checkpoint.checkpoint(self.vit_blocks, s, A_map,max_point)
#             # # s = self.vit_blocks(s, A_map, casename)
#             #
#             # s = self.mamba(s, A_map.unsqueeze(1), A_map.unsqueeze(1))
#             # #
#             # s_in = x.detach()
#             # # s_in = s_in.view(s_in.shape[0], s_in.shape[1], s_in.shape[2] * s_in.shape[3]).transpose(-1, -2)
#             # # A_map_in = A_map.view(A_map.shape[0], 1, A_map.shape[1], A_map.shape[2]).transpose(-1, -2)
#             # # s_c = self.assa(s_in, mask=A_map_in)
#             # # s_c = s_c.transpose(-1, -2)
#             # # s_c = s_c.view(s_c.shape[0], s_c.shape[1], s.shape[2], s.shape[3])
#             # #
#             # A_map_in = A_map.view(A_map.shape[0], 1, A_map.shape[1], A_map.shape[2]).transpose(-1, -2)
#             # s_c = self.assa(s_in, mask=A_map_in)
#             # #
#             # s = s + s_c
#             # # # print('xshape', A_map.shape)
#             # # # s = self.mamba(s,A_map.unsqueeze(1),A_map.unsqueeze(1))
#             # s = self.vit_blocks(s, A_map, casename)
#             # if self.deep_supervision_T is not None:
#             #     ds_T = self.deep_supervision_T(s)
#             #
#             # x = x + s
#             # print('x2:',x.shape)
#             # print('ds2:', x.shape)
#
#             # 无broadcast
#             # print('3333',self.bound_size)
#
#             pro = UnoB_fang(tar[:, 0].long().detach().cpu().numpy(), A_map.cpu().numpy(), self.bound_size)
#             pro_pool = UnoB_fang(tar_pool[:, 0].long().detach().cpu().numpy(), A_map_pool.cpu().numpy(), self.bound_size)
#             # print('4444',pro.shape)
#             A_map = torch.tensor(pro).cuda()
#             A_map_pool = torch.tensor(pro_pool).cuda()
#             # print('self.num_stage',A_map.shape)
#             # print('self.num_stage',s.shape)
#             s_shape = s.shape
#             s = self.mamba(s.reshape(s_shape[0], s_shape[1], s_shape[2]*s_shape[3]).transpose(-1, -2), A_map.unsqueeze(1), A_map.unsqueeze(1))
#             # print('self.num_stage',s.shape)
#             s_in = x.detach()
#             A_map_in = A_map.view(A_map.shape[0], 1, A_map.shape[1], A_map.shape[2])
#             A_map_in_pool = A_map_pool.view(A_map_pool.shape[0], 1, A_map_pool.shape[1], A_map_pool.shape[2])
#             # print(A_map_in.shape)
#             s_c = self.assa(s_in, mask=A_map_in, mask2 = A_map_in_pool)
#             s = s.transpose(-1, -2).reshape(s_shape[0], s_shape[1], s_shape[2], s_shape[3]) + s_c
#             s = self.vit_blocks(s, A_map, casename)
#             if self.deep_supervision_T is not None:
#                 ds_T = self.deep_supervision_T(s)
#             x = x + s
#
#             return x, ds, A_map, ds_T
#         else:
#             # x = self.deep_supervision_T(x)
#             # print('x3:',x.shape)
#             # print('ds3:', x.shape)
#             return x, ds, None, None


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.num_class = self.config.n_classes
        # print('num_class',self.num_class)

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]


        dmodels = [768, 768, 768, 768]
        depths = [1, 1, 1, 1]
        # num_heads = [3, 3, 3, 6]
        # patch_size = [[4, 4], [2, 2], [1, 1], [1, 1]]
        # bound_sizes = [9, 7, 3, 3]
        # patch_size = [[16, 16],[8,8],[4,4],[2,2]]
        dim_head = [64,64,64,64]
        pool_op_kernel_sizes = [8, 4, 2, 1]
        num_stage = [1,2,3,4]
        num_heads = [16, 8, 4, 2]
        patch_size = [[2, 2], [4, 4], [8, 8], [16, 16]]
        bound_sizes = [3, 3, 7, 9]

        input_resolution =  [[int(352/i),int(352/i)] for i in pool_op_kernel_sizes]
        blocks = [
            # DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            DecoderBlock_uct(in_channels=in_ch, out_channels=out_ch, skip_channels=sk_ch, dmodel=int(dm), depths=de,
                             input_resolution=ir,
                             num_heads=nh,
                             patch_size=ps,
                             dim_head=dh, num_stage=ns, bound_size=bs,num_class=self.num_class)
            for in_ch, out_ch, sk_ch, dm, de, ir, nh, ps, dh, ns, bs in
            zip(in_channels, out_channels, skip_channels, dmodels, depths, input_resolution, num_heads, patch_size,
                dim_head, num_stage, bound_sizes)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None, casename = '1'):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        out = []
        out_ds_T, out_Amap = [], []
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x, ds, A_map, ds_T = decoder_block(x, skip=skip, casename = casename)

            if ds_T is not None:
                out.append(ds)
                out_ds_T.append(ds_T)
                out_Amap.append(A_map)
                # print('A_MAP_SHAPE:', A_map.shape)
            else:
                out.append(ds)
            # print(out[::-1].shape)
        return x, out[::-1], out_ds_T[::-1], out_Amap[::-1]
        # return x, ds, A_map, ds_T


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=1024, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x, casename):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x, outputs, DS_T, UMap = self.decoder(x, features, casename = casename)
        logits = self.segmentation_head(x)
        # print('logits')
        # for item in logits:
        #     print(f"数据类型: {type(item)}{item.shape}")
        # print(outputs.append(logits[1]))
        # print('outputs')
        # for item in outputs:
        #     print(f"数据类型: {type(item)}{item.shape}")
        return logits, outputs, DS_T, UMap

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


