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
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        # self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.slots_mu = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.ones(1, num_slots, dim))

        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
    


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution, out_channels=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, out_channels+1, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionEncoder(nn.Module):
    def __init__(self, resolution=(28, 56), num_slots=3, num_iterations=3, hid_dim=64):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)

    def forward(self, image, num_slots=None):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        image = image.permute(0,3,1,2)
        x = self.encoder_cnn(image)  # CNN Backbone.

        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x, num_slots=num_slots)
        return slots
    
"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionDecoder(nn.Module):
    def __init__(self, resolution=(28, 56), num_slots=3, num_iterations=3, hid_dim=64):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

    def forward(self, slots, num_slots=None):
        if num_slots == None:
            num_slots = self.num_slots
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
        out_channel = 1
        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(slots.shape[0] // num_slots, -1, x.shape[1], x.shape[2], x.shape[3]).split([out_channel,1], dim=-1)
        

        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        return recon_combined.permute(0,2,3,1), recons