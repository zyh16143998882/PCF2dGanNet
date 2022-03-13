import functools

import torch
from torch import nn
import torch.nn.functional as F

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class UnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetEncoder, self).__init__()

        # Encoder layers
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf*2)

        self.e3_c = spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e3_norm = norm_layer(ngf*4)

        self.e4_c = spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e4_norm = norm_layer(ngf*8)

        self.e5_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e5_norm = norm_layer(ngf*8)

        self.e6_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e6_norm = norm_layer(ngf*8)

        self.e7_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e7_norm = norm_layer(ngf*8)



    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        # No norm on the inner_most layer
        e8 = F.sigmoid(e7)

        return e8

class UnetGanEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetGanEncoder, self).__init__()

        self.unet_encoder = UnetEncoder(1, 64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False),use_spectral_norm=False)
        # 结束输出是512*2*2
        self.gan_generator = EasyUnetGenerator(512,64,norm_layer=functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False),use_spectral_norm=False)
        # 用来计算图片feature域上的loss
        self.criterionL2_feature_loss = torch.nn.MSELoss()

    # In this case, we have very flexible unet construction mode.
    def forward(self, partial_img,gt_img):
        # Encoder
        # No norm on the first layer
        x = self.unet_encoder(partial_img)
        x = self.gan_generator(x)

        gt_x = self.unet_encoder(gt_img)
        # 计算feature loss并返回
        feature_loss = self.criterionL2_feature_loss(x,gt_x)

        return x,feature_loss

class EasyUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(EasyUnetGenerator, self).__init__()
        self.e1_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e2_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf * 8)
        self.e3_c = spectral_norm(nn.Conv2d(ngf*8, ngf*16, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e3_norm = norm_layer(ngf * 16)
        self.e4_c = spectral_norm(nn.Conv2d(ngf*16, ngf*16, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.d1_c = spectral_norm(nn.ConvTranspose2d(ngf * 16, ngf * 16, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d1_norm = norm_layer(ngf * 16)
        self.d2_c = spectral_norm(nn.ConvTranspose2d(ngf * 16 * 2, ngf * 8, kernel_size=3, stride=1, padding=1),use_spectral_norm)
        self.d2_norm = norm_layer(ngf * 8)
        self.d3_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=3, stride=1, padding=1),use_spectral_norm)
        self.d3_norm = norm_layer(ngf * 8)
        self.d4_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=3, stride=1, padding=1),use_spectral_norm)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_c(F.leaky_relu_(e3, negative_slope=0.2))
        # Decoder
        d1 = self.d1_norm(self.d1_c(F.relu_(e4)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e3], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e2], dim=1))))
        d4 = self.d4_c(F.relu_(torch.cat([d3, e1], 1)))

        d4 = F.sigmoid(d4)

        return d4