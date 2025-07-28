import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .FADC import OmniAttention, FrequencySelection, generate_laplacian_pyramid, AdaptiveDilatedConv
from  .FADConv import AdaptiveDilatedConv1


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


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block with AdaptiveDilatedConv1."""

    def __init__(self, cin, cout=None, cmid=None, stride=1, dilation=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        # 使用 AdaptiveDilatedConv1 替代 conv3x3
        self.conv2 = AdaptiveDilatedConv1(cmid, cmid, kernel_size=3, stride=stride, dilation=dilation, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))  # 使用 AdaptiveDilatedConv1
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet model with MDFA at each stage."""

    def __init__(self, block_units, width_factor, dilation=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width, dilation=dilation))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width, dilation=dilation)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2, dilation=dilation))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2, dilation=dilation)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2, dilation=dilation))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4, dilation=dilation)) for i in range(2, block_units[2] + 1)],
            ))),
        ]))

        # ✅ **在每个 Stage 之后添加 MDFA**
        self.attn_mdfa0 = MultiDimensionalFrequencyAttention(width)
        self.attn_mdfa1 = MultiDimensionalFrequencyAttention(width * 4)
        self.attn_mdfa2 = MultiDimensionalFrequencyAttention(width * 8)
        self.attn_mdfa3 = MultiDimensionalFrequencyAttention(width * 16)

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()

        # ✅ **Root 层处理**
        x = self.root(x)  # 经过 Root 层
        features.append(x)  # 🚀 把 `root` 作为第一个跳跃连接

        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        # ✅ **遍历 block1, block2, block3**
        for i in range(len(self.body)):
            x = self.body[i](x)  # **先经过 block**
            x = getattr(self, f'attn_mdfa{i + 1}')(x)  # **MDFA 作用在 `block` 之后**

            if i < 2:  # ✅ 只存储 `block1` 和 `block2`
                features.append(x)  # **MDFA 处理后的 `x` 作为 Skip Connection**

        return x, features  # **返回正常顺序的 `features`**




