import torch
import torch.nn as nn
import torch.nn.functional as F
from base_networks import *

import math

class InnerTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(InnerTransformer, self).__init__()
        self.linear_in = nn.Linear(in_dim, out_dim)  # Transform from in_dim to out_dim
        self.norm1 = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, int(out_dim * mlp_ratio)),  # This is correct
            nn.GELU(),
            nn.Linear(int(out_dim * mlp_ratio), out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.size()  # Get batch size, channels, height, and width
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # Reshape for LayerNorm and attention
        x = self.linear_in(x)  # Apply linear transformation to adjust dimensions from c to out_dim
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # Reshape back to original
        return x

class OuterTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(OuterTransformer, self).__init__()
        self.linear_in = nn.Linear(in_dim, out_dim)  # Transform from in_dim to out_dim
        self.norm1 = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, int(out_dim * mlp_ratio)),  # This is correct
            nn.GELU(),
            nn.Linear(int(out_dim * mlp_ratio), out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.size()  # Get batch size, channels, height, and width
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # Reshape for LayerNorm and attention
        x = self.linear_in(x)  # Apply linear transformation to adjust dimensions from c to out_dim
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # Reshape back to original
        return x

class TNTBlock(nn.Module):
    def __init__(self, in_dim, outer_dim, inner_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TNTBlock, self).__init__()
        self.outer_transformer = OuterTransformer(in_dim, outer_dim, num_heads, mlp_ratio, dropout)
        self.inner_transformer = InnerTransformer(outer_dim, inner_dim, num_heads, mlp_ratio, dropout)

    def forward(self, x):
        outer_output = self.outer_transformer(x)
        inner_output = self.inner_transformer(outer_output)
        return inner_output

class TransweatherTNT(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(TransweatherTNT, self).__init__()

        # Using fewer layers and reduced dimensions
        self.encoder = nn.Sequential(
            TNTBlock(in_dim=3, outer_dim=16, inner_dim=16, num_heads=2),
            TNTBlock(in_dim=16, outer_dim=32, inner_dim=32, num_heads=2),
        )

        self.decoder = nn.Sequential(
            TNTBlock(in_dim=32, outer_dim=32, inner_dim=32, num_heads=2),
            TNTBlock(in_dim=32, outer_dim=16, inner_dim=16, num_heads=2),
            TNTBlock(in_dim=16, outer_dim=8, inner_dim=8, num_heads=2)  # Reducing the output channels gradually
        )

        self.convtail = convprojection(reduced=True)
        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()

        if path is not None:
            self.load(path)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.convtail(x)
        clean = self.active(self.clean(x))
        return clean

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class convprojection(nn.Module):
    def __init__(self, path=None, reduced=False, **kwargs):
        super(convprojection, self).__init__()

        # Adjusting initial channels to align with previous outputs
        input_channels = 8 if reduced else 512  # Modify based on the actual output of the last encoder layer
        output_channels = 16 if reduced else 512
        mid_channels = 8 if reduced else 320
        small_channels = 4 if reduced else 128

        self.convd32x = UpsampleConvLayer(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
        self.convd16x = UpsampleConvLayer(output_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.dense_4 = nn.Sequential(ResidualBlock(mid_channels))
        self.convd8x = UpsampleConvLayer(mid_channels, small_channels, kernel_size=4, stride=2, padding=1)
        self.dense_3 = nn.Sequential(ResidualBlock(small_channels))
        self.convd4x = UpsampleConvLayer(small_channels, small_channels, kernel_size=4, stride=2, padding=1)
        self.dense_2 = nn.Sequential(ResidualBlock(small_channels))
        self.convd2x = UpsampleConvLayer(small_channels, small_channels, kernel_size=4, stride=2, padding=1)
        self.dense_1 = nn.Sequential(ResidualBlock(small_channels))
        self.convd1x = UpsampleConvLayer(small_channels, 4, kernel_size=4, stride=2, padding=1)
        self.conv_output = ConvLayer(4, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

    def forward(self, x):
        print("Input to convtail:", x.shape)

        res32x = self.convd32x(x)
        print("After convd32x:", res32x.shape)
        res16x = self.convd16x(res32x)
        print("After convd16x:", res16x.shape)

        res8x = self.convd8x(self.dense_4(res16x))
        print("After convd8x:", res8x.shape)
        res4x = self.convd4x(self.dense_3(res8x))
        print("After convd4x:", res4x.shape)
        res2x = self.convd2x(self.dense_2(res4x))
        print("After convd2x:", res2x.shape)
        x = self.convd1x(self.dense_1(res2x))
        print("After convd1x:", x.shape)

        output = self.conv_output(x)
        final_output = self.active(output)
        print("Final output:", final_output.shape)
        
        return final_output

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out
