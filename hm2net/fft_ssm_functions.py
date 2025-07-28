import torch.nn as nn
import torch
from mamba_ssm import Mamba
import math
import torch.nn.functional as F
from einops import rearrange, repeat

import pywt
from typing import Sequence, Tuple, Union, List
class FFTBlock_2d_noresize(nn.Module):  #Conventional하게 rff2d사용해서 H/W처리하는경우, 단 H방향으로 값이 줄어들게 설정: epoch좀만있으면 겨로가나올것으로 보임
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels,   kernel_size=3, stride=1, padding=1)
        self.conv_spectral = nn.Conv2d(channels*2, channels*2, kernel_size=1, stride=1, padding=0)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=channels,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x))
        
        # Mamba 처리
        b, c, h, w = x.shape
        # # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        # x_mamba = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        # x_mamba = self.mamba(x_mamba)
        # # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        # x_mamba = x_mamba.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 주파수 도메인 처리
        fft = torch.fft.rfft2(x, s=(w, h), dim=(3, 2), norm='ortho') # [2, 11, 2688 , 192] -> [2, 11, 2688, 97] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 11, 1345, 192] -> [2, 22, 1345, 192]
        
        fft_conv = F.relu(self.conv_spectral(fft_concat))
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft2(fft_complex, s=(w, h), dim=(3, 2), norm='ortho') # [2, 11, 384, 192] 
        
        # 모든 처리를 합침
        return spatial + spectral# + x_mamba
class FFTBlock_1d_conv1x1channel_noresize(nn.Module):  #1x1 conv가 channel수가 작아지니 초반에는 제대로 학습되나 뒤에 또 문제 (이전에는 너무 많음) - 2688이라 학습에 시간이 필요할수도
    def __init__(self, channels, mamba=False):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels,   kernel_size=3, stride=1, padding=1)
        self.conv_spectral = nn.Conv2d(channels*2, channels*2, kernel_size=1, stride=1, padding=0)
        
        # Mamba 모델 추가
        self.mamba_model = Mamba(
            d_model=channels*2,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.mamba = mamba
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x))
        
        # Mamba 처리
        b, c, h, w = x.shape
        # # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        # x_mamba = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        # x_mamba = self.mamba(x_mamba)
        # # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        # x_mamba = x_mamba.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 주파수 도메인 처리
        fft = torch.fft.rfft(x, n=h, dim=2, norm='ortho') # [2, 11, 2688 , 192] -> [2, 11,1345, 192] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 11, 1345, 192] -> [2, 22, 1345, 192]
        
        fft_conv = F.relu(self.conv_spectral(fft_concat))
        # Mamba 처리: width 방향으로 독립적으로 처리
        if self.mamba:
            # [B, C, H, W] -> [B*W, H, C] 형태로 변환
            fft_conv = rearrange(fft_conv, 'b c h w -> (b w) h c')
            fft_conv = self.mamba_model(fft_conv)
            # 다시 원래 형태로 변환
            fft_conv = rearrange(fft_conv, '(b w) h c -> b c h w', b=b)
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft(fft_complex, n=h, dim=2, norm='ortho') # [2, 11, 384, 192] 
        
        # 모든 처리를 합침
        return spatial + spectral# + x_mamba
class FFTBlock_1d_conv1x1width_resize(nn.Module): #결과 아예 안나옴 - 문제가됨
    def __init__(self, channels):
        super().__init__()
        # self.conv_spatial = nn.Conv2d(channels, channels,   kernel_size=3, stride=1, padding=1)
        self.conv_spatial = nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=(7, 1), padding=(0, 0))
        self.conv_spectral = nn.Conv2d(2690, 193*2, kernel_size=1, stride=1, padding=0)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=channels,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x))
        
        # Mamba 처리
        b, c, h, w = x.shape
        # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        x_mamba = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_mamba = self.mamba(x_mamba)
        # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        x_mamba = x_mamba.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 주파수 도메인 처리
        x = rearrange(x, 'b c h w -> b h c w')  # 

        # fft = torch.fft.rfft2(x, s=(c, w), dim=(2, 3), norm='ortho') #[2, 2688, 11, 97]
        # 주파수 도메인 처리
        fft = torch.fft.rfft(x, n=h, dim=1, norm='ortho') # [2, 2688, 11, 192] -> [2, 1345, 11, 192] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 1345, 11, 192] -> [2, 2690, 11, 192]
        
        fft_conv = F.relu(self.conv_spectral(fft_concat))   #[2, 2690, 11, 192] -> [2, 193*2, 11, 192]
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        # spectral = torch.fft.irfft2(fft_complex, s=(c, w), dim=(2, 3), norm='ortho')
        spectral = torch.fft.irfft(fft_complex, n=384, dim=1, norm='ortho') # [2, 11, 384, 192] 
        spectral = rearrange(spectral, 'b h c w -> b c h w')  # 
        
        # 모든 처리를 합침
        return spatial + spectral# + x_mamba
class FFTBlock_1d_conv1x1width_noresize(nn.Module): #결과잘안나옴 - 2688이라 학습에 시간이 필요할수도
    def __init__(self, channels, features=2688):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels,   kernel_size=3, stride=1, padding=1)
        self.conv_spectral = nn.Conv2d(int(features/2+1)*2, int(features/2+1)*2, kernel_size=1, stride=1, padding=0)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=channels,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x))
        
        # Mamba 처리
        b, c, h, w = x.shape
        # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        # x_mamba = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        # x_mamba = self.mamba(x_mamba)
        # # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        # x_mamba = x_mamba.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 주파수 도메인 처리
        x = rearrange(x, 'b c h w -> b h c w')  # 

        # fft = torch.fft.rfft2(x, s=(c, w), dim=(2, 3), norm='ortho') #[2, 2688, 11, 97]
        # 주파수 도메인 처리
        fft = torch.fft.rfft(x, n=h, dim=1, norm='ortho') # [2, 2688, 11, 192] -> [2, 1345, 11, 192] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 1345, 11, 192] -> [2, 2690, 11, 192]
        
        fft_conv = F.relu(self.conv_spectral(fft_concat))
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        # spectral = torch.fft.irfft2(fft_complex, s=(c, w), dim=(2, 3), norm='ortho')
        spectral = torch.fft.irfft(fft_complex, n=h, dim=1, norm='ortho') # [2, 11, 384, 192] 
        spectral = rearrange(spectral, 'b h c w -> b c h w')  # 
        
        # 모든 처리를 합침
        return spatial + spectral# + x_mamba
    
class FFTSSMBlock_1d_resize(nn.Module): #결과 이상하게 나옴 - 2688이라 학습에 시간이 필요할수도
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=(7, 1), padding=(0, 0))
        #self.conv_spectral = nn.Conv1d(4224, 4224, kernel_size=3, stride=4, padding=95)
        self.conv_spectral = nn.Conv1d(4224, 4224, kernel_size=7, stride=7, padding=3)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=193,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x)) #relu?
        
        # Mamba 처리
        b, c, h, w = x.shape
        # 주파수 도메인 처리
        fft = torch.fft.rfft(x, n=h, dim=2, norm='ortho') # [2, 11, 2688, 192] -> [2, 11, 1345, 192] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 11, 1345, 192] -> [2, 22, 1345, 192]
        
        fft_concat = rearrange(fft_concat, 'b c h w -> b (c w) h')  # [2, 22, 1345, 192] -> [2, 4224, 1345]

        fft_conv = F.relu(self.conv_spectral(fft_concat))       # [2, 4224, 1345] -> [2, 4224, 193] 
        
        # fft_conv = self.mamba(fft_conv)  # [2, 4224, 193]  -> [2, 4224, 193] 

        fft_conv = rearrange(fft_conv, ' b (c w) h -> b c h w', c=c*2, w=w)  # [2, 4224, 193] -> [2, 22, 193, 192]

        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)    # [2, 22, 193, 192] -> [2, 11, 193, 192]
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft(fft_complex, n=384, dim=2, norm='ortho') # [2, 11, 384, 192] 

        # 모든 처리를 합침
        return spatial + spectral
    
'''
class FFTSSMBlock2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=(7, 1), padding=(0, 0))
        #self.conv_spectral = nn.Conv1d(4224, 4224, kernel_size=3, stride=4, padding=95)
        self.conv_spectral = nn.Conv1d(4224, 4224, kernel_size=7, stride=7, padding=3)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=channels,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = F.relu(self.conv_spatial(x)) #relu?
        
        # Mamba 처리
        b, c, h, w = x.shape
        # 주파수 도메인 처리
        fft = torch.fft.rfft(x, n=h, dim=2, norm='ortho') # [2, 11, 2688, 192] -> [2, 11, 1345, 192] 
        # 주파수도메인 길이=(input_length // 2 + 1)
        fft_concat = torch.cat([fft.real, fft.imag], dim=1)    # [2, 11, 1345, 192] -> [2, 22, 1345, 192]
        
        fft_concat = rearrange(fft_concat, 'b c h w -> b (c w) h')  # [2, 22, 1345, 192] -> [2, 4224, 1345]

        fft_conv = F.relu(self.conv_spectral(fft_concat))       # [2, 4224, 1345] -> [2, 4224, 193] 
        
        fft_conv = rearrange(fft_conv, ' b (c w) h -> b c h w', c=c*2, w=w)  # [2, 4224, 193] -> [2, 22, 193, 192]

        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)    # [2, 22, 193, 192] -> [2, 11, 193, 192]
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft(fft_complex, n=384, dim=2, norm='ortho') # [2, 11, 384, 192] 
        
        # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        x_mamba = rearrange(x, 'b c h w -> b (h w) c')  # [B, C, H, W] -> [B, H*W, C]
        x_mamba = self.mamba(x_mamba)
        # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        x_mamba = rearrange(x_mamba, 'b (h w) c -> b c h w', h=h, w=w)  # [B, H*W, C] -> [B, C, H, W]

        # 모든 처리를 합침
        return spatial + spectral + x_mamba
'''
class FFTBlock(nn.Module): # Simpple version of FFT
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_spectral = nn.Conv2d(channels*2, channels*2, 1)
        
    def forward(self, x):
        # 공간 도메인 처리
        spatial = self.conv_spatial(x)
        
        # 주파수 도메인 처리 
        fft = torch.fft.rfft2(x)
        fft_real = fft.real
        fft_imag = fft.imag
        fft_concat = torch.cat([fft_real, fft_imag], dim=1)
        
        fft_conv = self.conv_spectral(fft_concat)
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft2(fft_complex)
        
        return spatial + spectral

class FFTSSMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_spectral = nn.Conv2d(channels*2, channels*2, 1)
        
        # Mamba 모델 추가
        self.mamba = Mamba(
            d_model=channels,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
        
    def forward(self, x):
        # 기존 FFT 처리
        spatial = self.conv_spatial(x) #relu?
        
        # Mamba 처리
        b, c, h, w = x.shape
        # [B, C, H, W] -> [B, H*W, C] 형태로 변환
        x_mamba = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_mamba = self.mamba(x_mamba)
        # 다시 원래 형태로 변환 [B, H*W, C] -> [B, C, H, W]
        x_mamba = x_mamba.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 주파수 도메인 처리
        fft = torch.fft.rfft2(x)
        fft_real = fft.real
        fft_imag = fft.imag
        fft_concat = torch.cat([fft_real, fft_imag], dim=1)
        
        fft_conv = self.conv_spectral(fft_concat) # relu?
        fft_real, fft_imag = torch.chunk(fft_conv, 2, dim=1)
        fft_complex = torch.complex(fft_real, fft_imag)
        spectral = torch.fft.irfft2(fft_complex)
        
        # 모든 처리를 합침
        return spatial + spectral + x_mamba

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, dim=2, mamba=None):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mamba = mamba
        self.dim = dim
        
        # Mamba 모델 추가
        self.mamba_model = Mamba(
            d_model=in_channels * 2,  # 채널 수를 d_model로 사용
            d_state=16,
            d_conv=4,
            expand=2
        )
    def forward(self, x):
        batch, c, h, w = x.size()
        
        if self.dim == 2:
            ffted = torch.fft.rfft2(x)
        elif self.dim == 1:
            ffted = torch.fft.rfft(x, n=h, dim=2, norm='ortho')

        ffted_real = ffted.real
        ffted_imag = ffted.imag
        ffted = torch.cat([ffted_real, ffted_imag], dim=1)
        
        # 주파수 도메인에서의 특징 추출
        ffted = self.conv_layer(ffted)
        ffted = self.bn(ffted)
        ffted = self.relu(ffted)
        
        # Mamba 처리: width 방향으로 독립적으로 처리
        if self.mamba:
            # [B, C, H, W] -> [B*W, H, C] 형태로 변환
            ffted = rearrange(ffted, 'b c h w -> (b w) h c')
            ffted = self.mamba_model(ffted)
            # 다시 원래 형태로 변환
            ffted = rearrange(ffted, '(b w) h c -> b c h w', b=batch)
        
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
        ffted_complex = torch.complex(ffted_real, ffted_imag)
        
        if self.dim == 2:
            output = torch.fft.irfft2(ffted_complex, s=(h, w))
        elif self.dim == 1:
            output = torch.fft.irfft(ffted_complex, n=h, dim=2, norm='ortho')

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, dim=2, mamba=None):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride != 1:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=stride)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, dim=dim, mamba=mamba)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups, dim=dim, mamba=mamba)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True, dim=2, mamba=None):
        super(FFC, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        if not (in_cg == 0 or out_cg == 0):
            assert stride == 1 or stride == 2 or stride == (1, 2) or stride == (2, 1), "Stride should be 1 or 2 or (1, 2) or (2, 1)." # This is because of FFT Calculation
        self.stride = stride
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, dim=dim, mamba=mamba)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
    
        return out_xl, out_xg


class FFCBlock(nn.Module):
    '''
    https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py#L5
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, comb=None, dim=2, mamba=None):
        super(FFCBlock, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, dim=dim, mamba=mamba)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        self.bn_l = lnorm(out_channels - int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
        self.comb = comb
    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        if self.comb is None:
            return x_l, x_g
        elif self.comb == 'local':
            return x_l
        elif self.comb == 'global':
            return x_g
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18/01/2023 4:28 am
# @Author  : Tianheng Qiu
# @FileName: waveelt_block.py
# @Software: PyCharm

"""
modified from ptwt, and also used pywt as a base
v1.0, 20230409
by thqiu
"""


def _as_wavelet(wavelet):
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def construct_2d_filt(lo, hi) -> torch.Tensor:
    """Construct two dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 2d filters of dimension
            [filt_no, 1, height, width].
            The four filters are ordered ll, lh, hl, hh.
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    # filt = filt.unsqueeze(1)
    return filt


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The length of the used filter.

    Returns:
        tuple: The numbers to attach on the edges of the input.

    """
    # pad to ensure we see all filter positions and for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are "reflect", "zero", "constant" and "periodic".
            Defaults to reflect.

    Returns:
        The padded output tensor.

    """

    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x

# global count
# count = 1
class LWN(nn.Module):
    def __init__(self, dim, wavelet='haar', initialize=True, use_ca=False, use_sa=False):
        super(LWN, self).__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*4, dim*6, 1)
        self.conv2 = nn.Conv2d(dim*6, dim*6, 7, padding=3, groups=dim*6)  # dw
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*6, dim*4, 1)
        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2),  # 上采样
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
            # self.sa_norm = LayerNorm2d(dim)
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True),  # conv2d
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

    def forward(self, x):
        _, _, H, W = x.shape
        ya, (yh, yv, yd) = self.wavedec(x) #WaveletConv
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x) #Conv 1x1
        x = self.conv2(x) #DWConv 7x7
        x = self.act(x) # GELU
        x = self.conv3(x) #Conv 1x1
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None) #Inverse WaveletConv
        if self.use_sa:
            sa_yh = self.sa_h(yh)
            sa_yv = self.sa_v(yv)
            y = y * (sa_yv + sa_yh)
        if self.use_ca:
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)
            ca_yv = self.ca_v(yv)
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # channel shuffle
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2   # gated channel attention
            y = y * ca
        return y

    def get_wavelet_loss(self):
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        """ Strang 107: Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation
        convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        """
        # polynomial multiplication is convolution, compute p(z):
        # print(dec_lo.shape, rec_lo.shape)
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0),
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero

    def alias_cancellation_loss(self):
        """ Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
            F0(z)H0(-z) + F1(z)H1(-z) = 0 """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi

        # # initial dec conv
        # self.conv = torch.nn.Conv2d(c1, c2 * 4, kernel_size=dec_filt.shape[-2:], groups=c1, stride=2)
        # self.conv.weight.data = dec_filt
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component




class WaveletUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, initialize=True, wavelet='haar'):
        # bn_layer not used
        super(WaveletUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels * 4,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 4)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)
        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)
    def forward(self, x):
        _, _, H, W = x.shape
        ya, (yh, yv, yd) = self.wavedec(x) #WaveletConv
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv_layer(dec_x)
        x = self.bn(x)
        x = self.relu(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        output = self.waverec([ya, (yh, yv, yd)], None) #Inverse WaveletConv

        return output

class WaveletTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lwa=True):
        # bn_layer not used
        super(WaveletTransform, self).__init__()
        self.enable_lwa = enable_lwa
        if stride != 1:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=stride)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.wa = WaveletUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lwa:
            self.lwa = WaveletUnit(
                out_channels // 2, out_channels // 2, groups)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.wa(x)

        if self.enable_lwa:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lwa(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class DFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True, enable_lwa=True, dim=2, mamba=None, direct_connect=False, kernel_size_lg=(-1,-1,-1,-1)):
        super(DFC, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        if not (in_cg == 0 or out_cg == 0):
            assert stride == 1 or stride == 2 or stride == (1, 2) or stride == (2, 1), "Stride should be 1 or 2 or (1, 2) or (2, 1)." # This is because of FFT Calculation
        self.stride = stride
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        if in_cg != 0 and out_cg != 0: 
            module = WaveletTransform 
            self.convl2l = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lwa)
        else:
            self.convl2l = module(in_cl, out_cl, kernel_size if kernel_size_lg[0] == -1 else kernel_size_lg[0],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size if kernel_size_lg[1] == -1 else kernel_size_lg[1],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size if kernel_size_lg[2] == -1 else kernel_size_lg[2],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, dim=dim, mamba=mamba)

        self.direct_connect = direct_connect
    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.direct_connect and (self.ratio_gout > 0 and self.ratio_gout < 1):
            out_xl = self.convl2l(x_l)
            out_xg = self.convg2g(x_g)
        else:
            if self.ratio_gout != 1:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            if self.ratio_gout != 0:
                out_xg = self.convl2g(x_l) + self.convg2g(x_g)
    
        return out_xl, out_xg


class DFCBlock(nn.Module):
    '''
    https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py#L5
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, enable_lwa=True, comb=None, dim=2, mamba=None, direct_connect=False, kernel_size_lg=(-1,-1,-1,-1)):
        super(DFCBlock, self).__init__()
        self.dfc = DFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, enable_lwa, dim=dim, mamba=mamba, direct_connect=direct_connect, kernel_size_lg=kernel_size_lg)  
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        self.bn_l = lnorm(out_channels - int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
        self.comb = comb
    def forward(self, x):
        x_l, x_g = self.dfc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        if self.comb is None:
            return x_l, x_g
        elif self.comb == 'local':
            return x_l
        elif self.comb == 'global':
            return x_g
        elif self.comb == 'concat':
            return torch.cat([x_l, x_g], dim=1)

            

class DFC_v2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True, enable_lwa=True, dim=2, mamba=None, direct_connect=False, kernel_size_lg=(-1,-1,-1,-1)):
        super(DFC_v2, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        if not (in_cg == 0 or out_cg == 0):
            assert stride == 1 or stride == 2 or stride == (1, 2) or stride == (2, 1), "Stride should be 1 or 2 or (1, 2) or (2, 1)." # This is because of FFT Calculation
        self.stride = stride
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size if kernel_size_lg[0] == -1 else kernel_size_lg[0],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size if kernel_size_lg[1] == -1 else kernel_size_lg[1],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        if in_cg != 0 and out_cg != 0: 
            module = WaveletTransform 
            self.convg2l = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lwa)
        else:
            self.convg2l = module(in_cg, out_cl, kernel_size if kernel_size_lg[2] == -1 else kernel_size_lg[2],
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, dim=dim, mamba=mamba)

        self.direct_connect = direct_connect
    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.direct_connect and (self.ratio_gout > 0 and self.ratio_gout < 1):
            out_xl = self.convl2l(x_l)
            out_xg = self.convg2g(x_g)
        else:
            if self.ratio_gout != 1:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            if self.ratio_gout != 0:
                out_xg = self.convl2g(x_l) + self.convg2g(x_g)
    
        return out_xl, out_xg


class DFCBlock_v2(nn.Module):
    '''
    https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py#L5
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, enable_lwa=True, comb=None, dim=2, mamba=None, direct_connect=False, kernel_size_lg=(-1,-1,-1,-1)):
        super(DFCBlock_v2, self).__init__()
        self.dfc = DFC_v2(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, enable_lwa, dim=dim, mamba=mamba, direct_connect=direct_connect, kernel_size_lg=kernel_size_lg)  
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        self.bn_l = lnorm(out_channels - int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
        self.comb = comb
    def forward(self, x):
        x_l, x_g = self.dfc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        if self.comb is None:
            return x_l, x_g
        elif self.comb == 'local':
            return x_l
        elif self.comb == 'global':
            return x_g
        elif self.comb == 'concat':
            return torch.cat([x_l, x_g], dim=1)

            