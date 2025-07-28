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
    """基于通道不确定性的 Embeddings 模块."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        if config.patches.get("grid") is not None:  # 采用 ResNet 作为 Hybrid Backbone
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = (config.patches["size"], config.patches["size"])
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

        x = self.patch_embeddings(x)  # (B, hidden_size, H', W')
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden_size)

        # 计算通道不确定性（标准差）
        uncertainty = torch.std(x, dim=2, keepdim=True)  # (B, n_patches, 1)

        # 重新排序 Token：按不确定性从低到高
        sorted_idx = torch.argsort(uncertainty, dim=1, descending=False)  # (B, n_patches, 1)

        # 确保索引尺寸匹配 position_embeddings
        sorted_idx = sorted_idx.expand(-1, -1, self.position_embeddings.shape[-1])  # (B, n_patches, hidden_size)

        # 重新排序 x
        x = torch.gather(x, dim=1, index=sorted_idx)

        # 生成动态 position embeddings
        sorted_position_embeddings = torch.gather(self.position_embeddings.repeat(x.shape[0], 1, 1),
                                                  dim=1, index=sorted_idx)

        # 计算最终 embeddings
        embeddings = x + sorted_position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features, sorted_idx



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

    def get_bidirectional_scanned_features(self, hidden_states, sorted_idx):
        """
        计算 4 种扫描方式的特征，并应用 sorted_idx
        """
        y_r1 = self.scan_forward(hidden_states, sorted_idx, mode="sequential")  # forward sequential
        y_r3 = self.scan_backward(hidden_states, sorted_idx, mode="sequential")  # backward sequential
        y_r2 = self.scan_forward(hidden_states, sorted_idx, mode="skip")  # forward skip
        y_r4 = self.scan_backward(hidden_states, sorted_idx, mode="skip")  # backward skip

        return y_r1, y_r3, y_r2, y_r4

    def scan_forward(self, hidden_states, sorted_idx, mode):
        """
        应用 sorted_idx 进行前向扫描，并添加轻微扰动
        """
        if mode == "sequential":
            sorted_idx = self.get_sorted_idx(hidden_states, sorted_idx, high_to_low=True)
        else:
            sorted_idx = self.get_skip_sorted_idx(hidden_states, sorted_idx, high_to_low=True)

        sorted_idx = sorted_idx.expand(-1, -1, hidden_states.shape[-1])
        out = torch.gather(hidden_states, dim=1, index=sorted_idx)

        # 🔹 轻微扰动，避免 scan_forward() 和 scan_backward() 过于相似
        out = out + 0.01 * torch.randn_like(out)

        # print(f"scan_forward mean: {out.mean().item():.6f}")  # 调试

        return out

    def scan_backward(self, hidden_states, sorted_idx, mode):
        """
        反向扫描：确保 sorted_idx 进行真正逆序排列，并且进行扰动
        """
        if mode == "sequential":
            sorted_idx = self.get_sorted_idx(hidden_states, sorted_idx, high_to_low=False)
        else:
            sorted_idx = self.get_skip_sorted_idx(hidden_states, sorted_idx, high_to_low=False)

        # 🔹 关键修改：确保 sorted_idx 逆序排列
        inv_sorted_idx = torch.argsort(sorted_idx, dim=1, descending=True)
        inv_sorted_idx = inv_sorted_idx.expand(-1, -1, hidden_states.shape[-1])

        out = torch.gather(hidden_states, dim=1, index=inv_sorted_idx)

        # 🔹 轻微扰动，防止 scan_forward() 和 scan_backward() 过于相似
        out = out + 0.01 * torch.randn_like(out)

        # print(f"scan_backward mean: {out.mean().item():.6f}")  # 调试

        return out

    def get_sorted_idx(self, hidden_states, sorted_idx, high_to_low=True):
        """
        计算 hidden_states 的像素级不确定性（通道方差），然后根据不确定性排序
        """
        uncertainty = torch.var(hidden_states, dim=-1, keepdim=True)

        # 🔹 **平滑 uncertainty，防止排序不稳定**
        uncertainty = torch.softmax(uncertainty, dim=1)

        sorted_idx = torch.argsort(uncertainty, dim=1, descending=high_to_low)

        return sorted_idx

    def get_skip_sorted_idx(self, hidden_states, sorted_idx, high_to_low=True):
        """
        计算基于 sorted_idx 的跳跃排序索引
        """
        sorted_idx = self.get_sorted_idx(hidden_states, sorted_idx, high_to_low)

        B, N, _ = sorted_idx.shape
        skip_idx = sorted_idx[:, ::2, :]  # 取偶数索引
        alt_idx = sorted_idx[:, 1::2, :]  # 取奇数索引

        # 🔹 **交替排列，而不是简单拼接**
        final_idx = torch.empty_like(sorted_idx)
        final_idx[:, 0::2, :] = skip_idx
        final_idx[:, 1::2, :] = alt_idx

        return final_idx








class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features, sorted_idx = self.embeddings(input_ids)  # 解析 3 个返回值
        encoded, attn_weights = self.encoder(embedding_output)  # 送入 Encoder
        return encoded, attn_weights, features

    # def get_bidirectional_features(self, input_ids):
    #     """
    #     获取前向 & 反向扫描的 4 个特征 y_r1, y_r3, y_r2, y_r4
    #     """
    #     embedding_output, features, sorted_idx = self.embeddings(input_ids)  # 确保解析 3 个返回值
    #     y_r1, y_r3, y_r2, y_r4 = self.encoder.get_bidirectional_scanned_features(embedding_output, sorted_idx)
    #     return y_r1, y_r3, y_r2, y_r4
    def get_bidirectional_features(self, input_ids):
        """
        获取前向 & 反向扫描的 4 个特征 y_r1, y_r3, y_r2, y_r4
        """
        embedding_output, features, sorted_idx = self.embeddings(input_ids)  # 确保解析 3 个返回值
        output = self.encoder.get_bidirectional_scanned_features(embedding_output, sorted_idx)

        # print(f"Encoder 输出长度: {len(output)}")  # 确保返回 4 个值
        return output  # 直接返回 output


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

        # conv1 处理拼接后的 x
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,  # 拼接 skip 后的通道数
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

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # x 先上采样

        # ✅ **增强模块（在 concat 之后）**
        if skip_channels > 0:
            self.enhanced_skip = EnhancedSkipConnection(in_channels + skip_channels)  # 增强输入通道

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)  # ✅ **Concat 之后增强**

        # 🚀 **增强仅在 concat 之后进行**
        if skip is not None:
            x = self.enhanced_skip(x)

        x = self.conv1(x)
        x = self.conv2(x)
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
        head_channels = 512  # ViT 头部输出通道数

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

        # ✅ 移除 self.enhanced_skip_connections，在 DecoderBlock 内部处理增强

        # ✅ final_conv 调整最终通道数
        self.final_conv = nn.Conv2d(64, 16, kernel_size=1)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        if features is not None:
            features = features[::-1]
            self.n_skip = min(self.n_skip, len(features))
            features = features[:self.n_skip]

        x = self.conv_more(x)  # ViT 输出变换到 head_channels

        for i, decoder_block in enumerate(self.blocks[:self.n_skip]):
            skip = features[i] if features is not None and i < len(features) else None
            x = decoder_block(x, skip=skip)  # ✅ skip 传入 DecoderBlock 内部增强

        x = self.final_conv(x)  # 🚀 变换通道数
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
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