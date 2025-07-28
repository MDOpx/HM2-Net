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
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .katransformer import KAN
import torch.nn.functional as F
from .FADC import OmniAttention, FrequencySelection, generate_laplacian_pyramid, AdaptiveDilatedConv
from .FADConv import AdaptiveDilatedConv1


logger = logging.getLogger(__name__)


class MultiDimensionalFrequencyAttention(nn.Module):
    """
    多维频率注意力机制（MDFA）+ 残差连接
    1. **通道注意力**: `OmniAttention` 计算不同通道的重要性
    2. **空间注意力**: `OmniAttention` 计算空间加权
    3. **频率注意力**: `FrequencySelection` 计算不同频率成分的加权
    4. **高低频特征分解**: `generate_laplacian_pyramid()` 进行高低频信息分解
    5. **频率掩码**: `AdaptiveDilatedConv.freq_select()` 进行高低频掩码增强
    6. **残差连接**: 直接加入 `x`，避免信息丢失
    """

    def __init__(self, in_channels, kernel_size=3, k_list=[2, 4, 8], lp_type='freq', use_residual=True):
        super(MultiDimensionalFrequencyAttention, self).__init__()
        self.in_channels = in_channels
        self.use_residual = use_residual  # 控制是否使用残差连接

        # **通道 & 空间注意力**
        self.omni_attention = OmniAttention(in_channels, in_channels, kernel_size)

        # **频率注意力**
        self.frequency_attention = FrequencySelection(in_channels, k_list=k_list, lp_type=lp_type)

        # **高低频分解**
        self.laplacian_pyramid_levels = len(k_list)

        # **频率掩码**
        self.freq_selector = AdaptiveDilatedConv(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x  # **残差连接**

        # **1. 计算 FFT 频谱**
        fft_x = torch.fft.fft2(x, norm='ortho')
        fft_x_real = fft_x.real
        fft_x_imag = fft_x.imag

        # **2. 计算通道注意力**
        channel_att = self.omni_attention.get_channel_attention(fft_x_real)

        # **🚀 这里修正 `channel_att` 的计算方式**
        if channel_att.shape[1] != C:
            # print(f"[DEBUG] 修正 channel_att 形状: {channel_att.shape} -> {(B, C, 1, 1)}")
            channel_att = channel_att.view(B, C, H, W)  # 重新调整形状
            channel_att = F.adaptive_avg_pool2d(channel_att, (1, 1))  # **确保是 (B, C, 1, 1)**

        # **3. 施加通道注意力**
        fft_x_real = fft_x_real * channel_att
        fft_x_imag = fft_x_imag * channel_att

        # **4. 计算空间注意力**
        spatial_att = self.omni_attention.get_spatial_attention(fft_x_real)
        fft_x_real = fft_x_real * spatial_att
        fft_x_imag = fft_x_imag * spatial_att

        # **5. 计算不同频率注意力**
        freq_weighted_x = self.frequency_attention(x)

        # **6. 计算高低频分解**
        pyramid = generate_laplacian_pyramid(x, self.laplacian_pyramid_levels)
        high_freq_part = sum(pyramid[:-1])  # 高频部分
        low_freq_part = pyramid[-1]  # 低频部分

        # **7. 计算频率掩码**
        freq_mask = self.freq_selector.freq_select(x)

        # **8. 结合所有注意力**
        fft_x_real = fft_x_real + freq_weighted_x
        fft_x_imag = fft_x_imag + freq_weighted_x
        fft_x_real = fft_x_real * freq_mask
        fft_x_imag = fft_x_imag * freq_mask

        # **9. 执行 IFFT 还原时域**
        fft_x_updated = torch.complex(fft_x_real, fft_x_imag)
        x_new = torch.fft.ifft2(fft_x_updated, norm='ortho').real

        # **10. 加入高低频信息**
        x_new = x_new + high_freq_part + low_freq_part

        # **11. 残差连接**
        if self.use_residual:
            x_new = x_new + residual  # 直接加上输入，增强稳定性

        return x_new


# =====================================
# 🔹 FastFourierConv2D 模块（局部 CNN + 频域全局特征）
# =====================================
class SpectralTransform2D(nn.Module):
    def __init__(self, in_channels):
        super(SpectralTransform2D, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),  # 频域信息处理
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def pad_to_power_of_two(self, x):
        """ 填充 H 和 W 维度到最近的 2 的幂次方 """
        B, C, H, W = x.shape
        H_new = 2 ** (H - 1).bit_length()
        W_new = 2 ** (W - 1).bit_length()
        pad_h = (0, H_new - H)
        pad_w = (0, W_new - W)
        x_padded = F.pad(x, pad=(*pad_w, *pad_h), mode="constant", value=0)
        return x_padded, (H, W)

    def crop_to_original_size(self, x, original_shape):
        """ 逆 FFT 之后裁剪回原始形状 """
        H, W = original_shape
        return x[:, :, :H, :W]

    def forward(self, x):
        residual = x  # **保存输入用于残差连接**

        # **前处理**
        x = self.pre_conv(x)

        # **填充到 2 的幂**
        x, original_shape = self.pad_to_power_of_two(x)
        x = x.float()

        # **FFT 计算**
        fft_out = torch.fft.rfft2(x, dim=(-2, -1))
        fft_real = fft_out.real.float()
        fft_imag = fft_out.imag.float()

        # **拼接实部 & 虚部**
        fft_features = torch.cat([fft_real, fft_imag], dim=1)

        # **频域信息卷积**
        fft_features = self.freq_conv(fft_features)

        # **拆分实部 & 虚部**
        split_idx = fft_features.size(1) // 2
        fft_real = fft_features[:, :split_idx, :, :].float()
        fft_imag = fft_features[:, split_idx:, :, :].float()

        fft_complex = torch.complex(fft_real, fft_imag)

        # **逆 FFT**
        global_out = torch.fft.irfft2(fft_complex, s=original_shape, dim=(-2, -1))
        global_out = self.crop_to_original_size(global_out, original_shape)

        # **调整通道数匹配 residual**
        if global_out.size(1) != residual.size(1):
            global_out = nn.Conv2d(global_out.size(1), residual.size(1), kernel_size=1).to(global_out.device)(global_out)

        # **残差连接**
        x = self.final_conv(global_out + residual)

        return x


class FastFourierConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, r=0.5):
        super(FastFourierConv2D, self).__init__()
        self.local_channels = int(r * in_channels)
        self.global_channels = in_channels - self.local_channels

        # **局部分支**
        self.local_branch1 = nn.Sequential(
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.ReLU(inplace=True)
        )
        self.local_branch2 = nn.Sequential(
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.ReLU(inplace=True)
        )

        # **全局分支**
        self.global_branch1 = nn.Sequential(
            nn.Conv2d(self.global_channels, self.global_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.global_channels),
            nn.ReLU(inplace=True)
        )
        self.global_branch2 = SpectralTransform2D(self.global_channels)

        # **最终融合**
        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(self.global_channels),
            nn.ReLU(inplace=True)
        )
        self.bn_relu2 = nn.Sequential(
            nn.BatchNorm2d(self.global_channels),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # **通道划分**
        local_features = x[:, :self.local_channels]
        global_features = x[:, self.local_channels:]

        # **局部分支 1**
        local_out1 = self.local_branch1(local_features)
        # **全局分支 1**
        global_out1 = self.global_branch1(global_features)
        fused_out1 = local_out1 + global_out1

        # **局部分支 2**
        local_out2 = self.local_branch2(local_features)
        # **全局分支 2**
        global_out2 = self.global_branch2(global_features)

        # **融合所有特征**
        fused_out2 = local_out2 + global_out2

        fused_out1 = self.bn_relu1(fused_out1)
        fused_out2 = self.bn_relu2(fused_out2)

        combined_out = torch.cat([fused_out1, fused_out2], dim=1)

        return self.output_conv(combined_out)

# =====================================
# 🔹 MFA 模块（使用多个 FastFourierConv2D 进行多尺度特征聚合）
# =====================================
class MFAFourier(nn.Module):
    """Multi-Scale Feature Aggregation with Fourier Transform"""
    def __init__(self, in_channels, out_channels, num_layers=9, r=0.5):
        super(MFAFourier, self).__init__()

        self.layers = nn.ModuleList([
            FastFourierConv2D(in_channels, out_channels, r=r) for _ in range(num_layers)
        ])

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 残差连接

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = self.residual(x)  # 残差分支
        for layer in self.layers:
            x = layer(x)  # 逐层传递 FastFourierConv2D 计算

        x = x + residual  # 残差相加
        return self.bn_relu(x)  # 归一化 + ReLU


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


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.kan = KAN(
            in_features=config.hidden_size,
            hidden_features=config.transformer["mlp_dim"],
            out_features=config.hidden_size,
            act_init="gelu",  # 你可以改成 "swish" 或其他 KAT 支持的激活方式
            drop=config.transformer["dropout_rate"]
        )

    def forward(self, x):
        return self.kan(x)


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

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # 1️⃣ 加载 Self-Attention 权重
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # 2️⃣ 加载 LayerNorm
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

            # 3️⃣ ⚠️ **跳过 MLP 权重加载**
            # print(f"Skipping MLP weights for Block {n_block}, using KAN instead")


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
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


class AdaptiveConv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
            fs_cfg=None,  # 频率选择参数
            kernel_decompose='both',  # 控制自适应卷积的模式
            use_dct=False,  # 是否使用DCT
    ):
        conv = AdaptiveDilatedConv1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=not use_batchnorm,
            fs_cfg=fs_cfg,
            kernel_decompose=kernel_decompose,
            use_dct=use_dct,
        )

        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
            super(AdaptiveConv2dReLU, self).__init__(conv, bn, relu)
        else:
            super(AdaptiveConv2dReLU, self).__init__(conv, relu)

class EnhancedSkipConnection(nn.Module):
    """
    🔹 改进的跳跃连接：
    1. **卷积-BN-ReLU** 预处理特征
    2. **FFT** 分解成 **实部 + 虚部**
    3. **通道注意力 (CA) + 空间注意力 (SA)**
    4. **IFFT 还原时域信息**
    5. **残差连接**
    6. **卷积-BN-ReLU** 进行最终特征融合
    7. **确保输入和输出通道数一致**
    """

    def __init__(self, channels, kernel_size=3):
        super(EnhancedSkipConnection, self).__init__()

        # 1️⃣ 预处理: Conv-BN-ReLU
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 2️⃣ 通道注意力 (CA)
        self.channel_attention = ChannelAttention(channels)

        # 3️⃣ 空间注意力 (SA)
        self.spatial_attention = SpatialAttention()

        # 4️⃣ 后处理: Conv-BN-ReLU
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(f"🟢【EnhancedSkipConnection】输入尺寸: {x.shape}")

        residual = x  # 保存原始输入用于残差连接

        # 1️⃣ 预处理
        x = self.pre_conv(x)
        # print(f"🔵【EnhancedSkipConnection】预处理后尺寸: {x.shape}")

        # 2️⃣ 进行 FFT 变换
        fft_x = torch.fft.fft2(x, norm='ortho')
        fft_real, fft_imag = fft_x.real, fft_x.imag
        # print(f"🔵【EnhancedSkipConnection】FFT 变换后尺寸: {fft_real.shape}")

        # 3️⃣ 分别对实部和虚部进行通道 & 空间注意力
        fft_real = self.channel_attention(fft_real)  # ✅ 直接输入，无需 unsqueeze
        fft_real = self.spatial_attention(fft_real)

        fft_imag = self.channel_attention(fft_imag)  # ✅ 直接输入
        fft_imag = self.spatial_attention(fft_imag)

        # print(f"🔵【EnhancedSkipConnection】通道 + 空间注意力后尺寸 (Real): {fft_real.shape}")
        # print(f"🔵【EnhancedSkipConnection】通道 + 空间注意力后尺寸 (Imag): {fft_imag.shape}")

        # 4️⃣ 还原到时域 (IFFT)
        fft_x_updated = torch.complex(fft_real, fft_imag)
        x_new = torch.fft.ifft2(fft_x_updated, norm='ortho').real
        # print(f"🔵【EnhancedSkipConnection】IFFT 还原后尺寸: {x_new.shape}")

        # 5️⃣ 残差连接
        x_new = x_new + residual

        # 6️⃣ 后处理
        x_new = self.post_conv(x_new)
        # print(f"🟢【EnhancedSkipConnection】输出尺寸: {x_new.shape}")

        return x_new


class ChannelAttention(nn.Module):
    """ Squeeze-and-Excitation (SE) 通道注意力 """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 计算均值，输出 (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"🔵【ChannelAttention】输入尺寸: {x.shape}")
        ca_weights = self.fc(self.avg_pool(x))  # (B, C, 1, 1) -> (B, C, 1, 1)
        # print(f"🔵【ChannelAttention】CA 权重尺寸: {ca_weights.shape}")
        return x * ca_weights  # **确保通道数不变**



class SpatialAttention(nn.Module):
    """ 空间注意力 (SA) """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_weights = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(sa_weights)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.skip_channels = skip_channels

        # `conv1` 处理拼接后的 `x`
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

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # ✅ `x` 先上采样

    def forward(self, x, skip=None):
        # print(f"🟠【DecoderBlock】输入 x 尺寸: {x.shape}")

        x = self.up(x)
        # print(f"🟠【DecoderBlock】上采样后 x 尺寸: {x.shape}")

        if skip is not None:
            # print(f"🔷【DecoderBlock】Skip 连接前的尺寸: {skip.shape}")

            # ✅ 确保 `skip` 和 `x` 形状匹配
            if x.shape[2:] != skip.shape[2:]:
                # print(f"⚠️【DecoderBlock】调整 Skip 尺寸: {skip.shape[2:]} → {x.shape[2:]}")
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            # print(f"🟢【DecoderBlock】拼接后 x 尺寸: {x.shape}")

        x = self.conv1(x)
        x = self.conv2(x)
        # print(f"🟣【DecoderBlock】Decoder 输出尺寸: {x.shape}")
        return x



class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)






class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512  # 设定 `ViT` 头部输出的通道数

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

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            self.n_skip = min(self.config.n_skip, len(skip_channels))
        else:
            skip_channels = [0, 0, 0, 0]
            self.n_skip = 0

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])

        self.enhanced_skip_connections = nn.ModuleList([
            EnhancedSkipConnection(skip_ch) if skip_ch > 0 else None for skip_ch in skip_channels
        ])

        # ✅ `final_conv` 调整最终通道数
        self.final_conv = nn.Conv2d(64, 16, kernel_size=1)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        # print(f"🟢【DecoderCup】解码器输入前的尺寸: {x.shape}")

        if features is not None:
            # print(f"🔵【DecoderCup】原始 features 长度: {len(features)}")
            # for i, f in enumerate(features):
            #     print(f"🔵【DecoderCup】Feature {i} 尺寸: {f.shape}")

            # 🚀 **确保 `skip0` 正确对应 `Feature2`**
            features = features[::-1]
            self.n_skip = min(self.n_skip, len(features))
            features = features[:self.n_skip]
            # print(f"🔵【DecoderCup】修正后 features 长度: {len(features)}")

        # print(f"🔵【DecoderCup】blocks 长度: {len(self.blocks)}")

        x = self.conv_more(x)  # `ViT` 输出变换到 `head_channels`
        # print(f"🟢【DecoderCup】经过 `conv_more` 变换后尺寸: {x.shape}")

        for i, decoder_block in enumerate(self.blocks[:self.n_skip]):
            if features is not None and i < len(features):
                skip = features[i]
            else:
                skip = None

            # if skip is not None:
            #     print(f"🔵【DecoderCup】Skip {i} 原始尺寸: {skip.shape}")

            if skip is not None and self.enhanced_skip_connections[i] is not None:
                expected_channels = self.enhanced_skip_connections[i].pre_conv[0].in_channels
                if skip.shape[1] != expected_channels:
                    # print(f"⚠️【DecoderCup】Skip {i} 通道不匹配: {skip.shape[1]} → {expected_channels}")
                    skip = nn.Conv2d(skip.shape[1], expected_channels, kernel_size=1).to(skip.device)(skip)

                skip = self.enhanced_skip_connections[i](skip)
                # print(f"🔵【DecoderCup】增强后的 Skip {i} 尺寸: {skip.shape}")

            # print(f"🟠【DecoderCup】当前 x 尺寸: {x.shape}，执行 `DecoderBlock`")
            x = decoder_block(x, skip=skip)

        x = self.final_conv(x)  # 🚀 变换通道数
        # print(f"🟢【DecoderCup】最终输出尺寸: {x.shape}")

        # ✅ **上采样到目标尺寸**
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # print(f"🟢【DecoderCup】上采样到 {x.shape}")

        return x







class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
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

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)

        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

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


