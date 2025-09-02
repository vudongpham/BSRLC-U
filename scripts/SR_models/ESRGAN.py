import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1):
        super().__init__()
        padding = 1 if stride > 1 else 'same'
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

    def forward(self, x):
        return self.conv(x)


class ResidualDenseBlock(nn.Module):
    """
    5‑layer dense block with growth channels (G) and residual scaling.
    """
    def __init__(self,
                 in_channel: int,
                 growth: int = 32,
                 scale: float = 0.2):
        super().__init__()
        self.scale = scale
        self.prelu = nn.PReLU()

        self.c1 = Conv2d(in_channel, growth, 3)
        self.c2 = Conv2d(in_channel + growth, growth, 3)
        self.c3 = Conv2d(in_channel + 2 * growth, growth, 3)
        self.c4 = Conv2d(in_channel + 3 * growth, growth, 3)
        self.c5 = Conv2d(in_channel + 4 * growth, in_channel, 3)

    def forward(self, x):
        x1 = self.prelu(self.c1(x))
        x2 = self.prelu(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.prelu(self.c3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.prelu(self.c4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.scale    # local residual with scaling


class RRDB(nn.Module):
    """
    Residual‑in‑Residual Dense Block composed of 3 RDBs.
    """
    def __init__(self,
                 in_channel: int,
                 growth: int = 32,
                 scale_rdb: float = 0.2,
                 scale_rrdb: float = 0.2):
        super().__init__()
        self.scale_rrdb = scale_rrdb
        self.rdb1 = ResidualDenseBlock(in_channel, growth, scale_rdb)
        self.rdb2 = ResidualDenseBlock(in_channel, growth, scale_rdb)
        self.rdb3 = ResidualDenseBlock(in_channel, growth, scale_rdb)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale_rrdb   # global residual of RRDB


class ESRGAN(nn.Module):
    def __init__(self,
                 in_channels: int = 6,
                 features: int = 64,
                 num_rrdb: int = 16,
                 scale: int = 3):
        super().__init__()

        self.input_channels = in_channels
        self.default_features = features
        self.scale = scale

        # Initial conv
        self.initial_conv = Conv2d(in_channels, features, kernel_size=3)

        # Trunk of RRDBs
        rrdb_blocks = [RRDB(features) for _ in range(num_rrdb)]
        self.trunk = nn.Sequential(*rrdb_blocks)

        # Conv after trunk
        self.trunk_conv = Conv2d(features, features, kernel_size=3)

        # Upsampling layers
        up_layers = []
 
        up_layers += [
                Conv2d(features, features * 9, kernel_size=3),
                nn.PixelShuffle(3),
                nn.PReLU(num_parameters=features, init=0.25),
            ]

        self.upsampling = nn.Sequential(*up_layers)

        # Final conv
        self.final_conv = Conv2d(features, in_channels, kernel_size=3)

    def forward(self, x_lr):
        x = torch.permute(x_lr, dims=(0, 3, 1, 2))     # NHWC → NCHW

        fea = self.initial_conv(x)
        trunk_out = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk_out                          # long skip over trunk

        fea = self.upsampling(fea)
        x_hr = self.final_conv(fea)

        x_hr = torch.permute(x_hr, dims=(0, 2, 3, 1))  # NCHW → NHWC
        return x_hr
