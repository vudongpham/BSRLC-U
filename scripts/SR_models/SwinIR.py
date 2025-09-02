import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    @staticmethod
    def _build_attn_mask(H, W, window_size, shift_size, device):
        if shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        # x: [B, L, C], where L = H*W (unknown a priori)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert H * W == L, "L must be a perfect square or provide H,W explicitly."

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        shift = min(H, W) >= self.window_size
        shift_size = self.shift_size if shift else 0
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self._build_attn_mask(H, W, self.window_size, shift_size, x.device)

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        x_windows = self.attn(x_windows, mask=attn_mask)
        x = window_reverse(x_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                 attn_drop=attn_drop, drop_path=drop_path)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x



class SwinIR(nn.Module):
    """
    SwinIR model for image restoration with configurable upscaling and arbitrary input/output channels.
    """
    def __init__(
        self,
        in_chans=6,
        out_chans=6,
        embed_dim=60,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        upscale=3
    ):
        super().__init__()
        self.upscale = upscale
        out_chans = out_chans or in_chans

        # shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # deep feature extraction layers
        self.layers = nn.ModuleList()
        # input_resolution = (img_size, img_size)
        for i in range(len(depths)):
            layer = BasicLayer(dim=embed_dim, depth=depths[i], num_heads=num_heads[i],
                   window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True,
                   drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)

        # reconstruction for upscaling
        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, out_chans, 3, 1, 1)
        )

    def forward(self, x_in):
        # Expect BHWC
        x = torch.permute(x_in, dims=(0, 3, 1, 2))  # B C H W
        x = self.conv_first(x)
        x_shallow = x  # <— cache shallow features

        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W

        x = self.conv_body(x)
        x = x + x_shallow            # <— global residual (important)
        x = self.upsample(x)

        x_hr = torch.permute(x, dims=(0, 2, 3, 1))  # return BHWC
        return x_hr


