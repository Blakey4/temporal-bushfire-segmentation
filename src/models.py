"""U-Net for burnt-area segmentation.

M1 (uni-temporal) = UNet(9): post-fire image only.
M2 (bi-temporal, early fusion) = UNet(18): pre- and post-fire images stacked as channels.
The two are identical except for the number of input channels of the very first convolution.

Architecture (depth 4, base width b; the final runs use b = 64):
    encoder     b -> 2b -> 4b -> 8b channels, each block followed by 2x2 max-pooling
    bottleneck  16b channels at 1/16 resolution (16x16 for a 256x256 patch)
    decoder     2x2 transposed conv (upsample), concatenate the encoder skip, block; back to b channels
    head        1x1 conv -> 1 channel of logits (probability of "burnt by this fire" = sigmoid(logit))
"""
import torch
from torch import nn


def norm_layer(kind, channels):
    """GroupNorm with 8 groups ("group") or BatchNorm ("batch")."""
    if kind == "group":
        return nn.GroupNorm(num_groups=8, num_channels=channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"unknown norm: {kind}")


def conv_block(in_channels, out_channels, norm="group"):
    """The plain U-Net block: (3x3 conv -> norm -> ReLU) twice. Bias is dropped as the norm has its own."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        norm_layer(norm, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        norm_layer(norm, out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    """Residual version of the block: ReLU(F(x) + skip(x)).

    F = conv-norm-ReLU-conv-norm; skip is the identity, or a 1x1 conv + norm when the channel
    count changes. The skip path lets gradients bypass the convolutions, which usually makes
    deeper or wider networks easier to optimise.
    """

    def __init__(self, in_channels, out_channels, norm="group"):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(norm, out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), norm_layer(norm, out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))


def make_block(in_channels, out_channels, norm="group", block="plain"):
    """One U-Net block: "plain" (conv_block) or "residual" (ResidualBlock)."""
    if block == "plain":
        return conv_block(in_channels, out_channels, norm)
    if block == "residual":
        return ResidualBlock(in_channels, out_channels, norm)
    raise ValueError(f"unknown block: {block}")


class UNet(nn.Module):
    """U-Net with `in_channels` inputs (9 = uni, 18 = bi) and one output channel of logits."""

    def __init__(self, in_channels, base=32, depth=4, norm="group", block="plain"):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]  # base 64: [64, 128, 256, 512, 1024]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths[:-1]:
            self.encoders.append(make_block(channels, width, norm, block))
            channels = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = make_block(widths[-2], widths[-1], norm, block)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2))
            self.decoders.append(make_block(width * 2, width, norm, block))  # skip + upsampled
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
