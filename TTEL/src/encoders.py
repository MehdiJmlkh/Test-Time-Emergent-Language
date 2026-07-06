import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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
            ConvBlock(hidden_size, hidden_size),
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        return x.reshape(x.shape[0], -1)


class IdentityEncoder(nn.Module):
    def __init__(self, representation_dim=2048):
        super().__init__()
        self.bn = nn.BatchNorm1d(representation_dim)

    def forward(self, x):
        x = self.bn(x)
        return x
