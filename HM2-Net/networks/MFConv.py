import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from networks.UltraLight_VM_UNet import PVMLayer_bing


class MambaLayer(nn.Module):
    def __init__(self, input_channels, d_state=16, expand=2):
        super(MambaLayer, self).__init__()
        self.input_channels = input_channels
        self.num_splits = 4
        assert input_channels % self.num_splits == 0, "输入通道数必须能被 4 整除"

        self.norm = nn.LayerNorm(input_channels)

        # self.mamba_layers = nn.ModuleList([
        #     Mamba(
        #         d_model=input_channels // self.num_splits,
        #         d_state=d_state // self.num_splits,
        #         d_conv=3,
        #         expand=expand
        #     )
        #     for _ in range(self.num_splits)
        # ])

        self.mamba_layers = nn.ModuleList([
            PVMLayer_bing(
                input_channels // self.num_splits,
                input_channels // self.num_splits
            )
            for _ in range(self.num_splits)
        ])

        self.projection = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.bn_output = nn.BatchNorm2d(input_channels)
        self.GELU_output = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        # print(1)
        x_flat = x.view(B, C, -1).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        parts = torch.chunk(x_norm, self.num_splits, dim=-1)
        processed_parts = [mamba(part) for mamba, part in zip(self.mamba_layers, parts)]
        x_mamba = torch.cat(processed_parts, dim=-1)
        out = x_mamba.transpose(-1, -2).reshape(B, C, H, W)

        out = self.projection(out)
        out = self.bn_output(out)
        out = self.GELU_output(out)

        return out



class SpectralTransform(nn.Module):
    def __init__(self, input_channels):
        super(SpectralTransform, self).__init__()

        self.conv_input = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.bn_input = nn.BatchNorm2d(input_channels)
        self.gelu_input = nn.GELU()

        self.conv_fft = nn.Conv2d(2 * input_channels, 2 * input_channels, kernel_size=1)
        self.bn_fft = nn.BatchNorm2d(2 * input_channels)
        self.gelu_fft = nn.GELU()

        self.conv_output = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.bn_output = nn.BatchNorm2d(input_channels)
        self.gelu_output = nn.GELU()

    def forward(self, x):
        x_skip = x

        x1 = self.gelu_input(self.bn_input(self.conv_input(x)))
        fft_result = torch.fft.rfft2(x1, dim=(2, 3), norm="ortho")
        fft_real, fft_imag = fft_result.real, fft_result.imag

        fft_concat = torch.cat([fft_real, fft_imag], dim=1)
        fft_conv = self.gelu_fft(self.bn_fft(self.conv_fft(fft_concat)))

        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        x_inv = torch.fft.irfft2(fft_complex, s=x.shape[-2:], dim=(2, 3), norm="ortho")

        x_res = x_skip + x_inv

        out = self.gelu_output(self.bn_output(self.conv_output(x_res)))

        return out





class MFConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFConv, self).__init__()
        self.local_channels = in_channels // 2
        self.global_channels = in_channels // 2

        # Local 分支
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.GELU(),
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.GELU(),
        )

        self.local_conv2 = nn.Sequential(
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.GELU(),
            nn.Conv2d(self.local_channels, self.local_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.local_channels),
            nn.GELU(),
        )

        # Global 分支
        self.global_inception = MambaLayer(self.global_channels, self.global_channels)
        self.global_spectral_transform = SpectralTransform(self.global_channels)

        # Global 归一化
        self.bn_global1 = nn.BatchNorm2d(self.global_channels)
        self.bn_global2 = nn.BatchNorm2d(self.global_channels)

        # 1x1 卷积用于融合通道
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.res_bn_relu = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


        # ✅ **在 `__init__()` 方法中初始化 `conv1x1`**
        self.initialize_weights()

    def initialize_weights(self):
        """ 确保 `conv1x1` 的初始化不会导致数值过大 """
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, x):
        res_x = self.res_conv(x)
        res_x = self.res_bn_relu(res_x)
        local, global_ = x.split(x.size(1) // 2, dim=1)

        # Local 处理
        local1 = self.local_conv1(local)
        local2 = self.local_conv2(local)

        # Global 处理
        global1 = self.global_inception(global_)
        global2 = self.global_spectral_transform(global_)

        # ✅ 归一化 local1, local2, global1, global2
        local1 = F.normalize(local1, p=2, dim=1)
        local2 = F.normalize(local2, p=2, dim=1)
        global1 = F.normalize(global1, p=2, dim=1)
        global2 = F.normalize(global2, p=2, dim=1)

        # **🔹 打印 `local` 和 `global` 结果的数值范围**
        # print(f"[MFConv] local1 -> min: {local1.min().item():.6f}, max: {local1.max().item():.6f}")
        # print(f"[MFConv] local2 -> min: {local2.min().item():.6f}, max: {local2.max().item():.6f}")
        # print(f"[MFConv] global1 -> min: {global1.min().item():.6f}, max: {global1.max().item():.6f}")
        # print(f"[MFConv] global2 -> min: {global2.min().item():.6f}, max: {global2.max().item():.6f}")

        # 进行融合
        fused1 = local1 + global1
        fused2 = local2 + global2

        # 拼接通道
        out = torch.cat([fused1, fused2], dim=1)
        out = out + res_x  # **残差连接前 `BatchNorm`**
        # print(f"[MFConv] res_x -> min: {res_x.min().item():.6f}, max: {res_x.max().item():.6f}")
        # print(f"[MFConv] out0 -> min: {out.min().item():.6f}, max: {out.max().item():.6f}")
        out = self.conv1x1(out)
        out = self.bn_relu(out)

        out = F.normalize(out, p=2, dim=1)*10  #

        # **🔹 打印 `out` 数值范围**
        # print(f"[MFConv] out -> min: {out.min().item():.6f}, max: {out.max().item():.6f}")

        return out





