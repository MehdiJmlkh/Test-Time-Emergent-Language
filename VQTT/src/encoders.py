import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.sequence(x)
    
class ProtoNetCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size)
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        return x.reshape(x.shape[0], -1)


class Dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(384, 384)
        # self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.head(x)
        # x = self.tanh(x)
        return x


class IdentityEncoder(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(2048)
    
    def forward(self, x):
        # if self.batch_norm:
        x = self.bn(x)
        return x

class ConvDecoderBlock(nn.Module):
    """
    Reverse of ConvBlock: Conv -> BN -> ReLU
    Upsampling handled outside to mirror MaxPool2d(2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.sequence(x)


class ProtoNetCNNDecoder(nn.Module):
    def __init__(self, representation_dim=1024, latent_channels=64, out_channels=3, upsample_times=4, sizes=None):
        super().__init__()
        
        # Channels mirror encoder (all 64 in your example)
        channels = [latent_channels] * upsample_times
        self.latent_channels = latent_channels
        self.sizes = sizes
        
        self.fc = nn.Linear(representation_dim, representation_dim)

        self.up_blocks = nn.ModuleList()
        for i in range(upsample_times):
            self.up_blocks.append(ConvDecoderBlock(channels[i], channels[i]))

        # Final conv to output channels (RGB)
        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.fc(x)

        if self.sizes is None:
            H_W = int((x.shape[1] // self.latent_channels) ** 0.5)
            x = x.view(x.shape[0], self.latent_channels, H_W, H_W)
            
            for block in self.up_blocks:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = block(x)
        else:
            x = x.view(x.shape[0], self.latent_channels, self.sizes[0][0], self.sizes[0][1])
            
            for block, (h, w) in zip(self.up_blocks, self.sizes[1:]):
                x = F.interpolate(x, size=(h, w), mode='nearest')
                x = block(x)

        x = torch.sigmoid(self.final_conv(x))
        return x.permute(0, 2, 3, 1)
    