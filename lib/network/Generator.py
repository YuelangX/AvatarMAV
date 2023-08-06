import torch
from torch import nn
from torch.nn import functional as F


class GeneratorBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_dim),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Generator(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.latent_dim = dims[0]

        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, dims[1], 4, bias=False),
            nn.Conv2d(dims[1], dims[1], 3, padding=1)
        )

        blocks = []
        for i in range(1, len(dims)-1):
            blocks.append(GeneratorBlock(dims[i], dims[i+1]))
        self.blocks = nn.Sequential(*blocks)

        self.final_block = nn.Conv2d(dims[-1], dims[-1], 3, padding=1)

    def forward(self, code):
        x = code[:, :, None, None]
        x = self.init_block(x)
        x = self.blocks(x)
        x = self.final_block(x)
        return x
