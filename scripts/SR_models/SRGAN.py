import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super().__init__()
        if stride > 1:
            padding = 1
        else:
            padding = 'same'

        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Residual_block(nn.Module):
    def __init__(self, in_channel, features_num):
        super().__init__()
        self.conv1 = Conv2d(in_channel, features_num, 3)
        self.prelu = nn.PReLU(num_parameters=features_num, init=0.25)
        self.conv2 = Conv2d(features_num, features_num, 3)
    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.prelu(x)
        x = self.conv2(x)
        x_out = x + x_in
        return x_out 




class SRGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 6
        self.default_features = 64
        self.scale = 3
        self.initial_conv = nn.Sequential(
            Conv2d(self.input_channels, self.default_features, kernel_size=9),
            nn.PReLU(num_parameters=self.default_features, init=0.25)
        )

        self.first_block = Residual_block(self.default_features, self.default_features)
        self.blocks = nn.Sequential(
            *[Residual_block(self.default_features, self.default_features) for _ in range(15)]
        )
        self.end_conv = nn.Sequential(
            Conv2d(self.default_features, self.default_features, kernel_size=3),
            nn.PReLU(num_parameters=self.default_features, init=0.25)
        )

        self.upsampling = nn.Sequential(
            Conv2d(self.default_features, self.default_features*(self.scale**2), kernel_size=3),
            nn.PixelShuffle(upscale_factor=self.scale),
            nn.PReLU(num_parameters=self.default_features, init=0.25)
        )

        self.final_conv = Conv2d(self.default_features, self.input_channels, kernel_size=9)

    def forward(self, x_lr):
        x = torch.permute(x_lr, dims=(0, 3, 1, 2))
        x_1 = self.initial_conv(x)
        x = self.first_block(x_1)
        x = self.blocks(x)
        x = self.end_conv(x)
        x = x + x_1
        x = self.upsampling(x)
        x = self.final_conv(x)
        x_hr = torch.permute(x, dims=(0, 2, 3, 1))
        return x_hr

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_feature = 6
        base_feature = 64
        self.conv_block = nn.Sequential(
            Conv2d(in_feature, base_feature, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.2),

            Conv2d(base_feature, base_feature, kernel_size=4, stride=2),
            nn.BatchNorm2d(base_feature),
            nn.LeakyReLU(negative_slope=0.2),

            Conv2d(base_feature, base_feature*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(base_feature*2),
            nn.LeakyReLU(negative_slope=0.2),

            Conv2d(base_feature*2, base_feature*4, kernel_size=4, stride=2),
            nn.BatchNorm2d(base_feature*4),
            nn.LeakyReLU(negative_slope=0.2),

            Conv2d(base_feature*4, base_feature*8, kernel_size=4, stride=2),
            nn.BatchNorm2d(base_feature*8),
            nn.LeakyReLU(negative_slope=0.2),

            Conv2d(base_feature*8, 1, kernel_size=4)
        )
    
    def forward(self, x_in):
        x = torch.permute(x_in, dims=(0, 3, 1, 2))
        x_out = self.conv_block(x)
        return x_out
        

            

