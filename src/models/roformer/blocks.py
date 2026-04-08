import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.roformer.layers import SplitAndMergeModule, LightRoformerInitialStage
from src.models.roformer.attention import InterleavedRoPEBlock

class EncoderBlock(nn.Module):
    """Block constituting one layer of the encoder (SMM + Downsampling)"""
    def __init__(self, n_band, n_split, in_G, out_G):
        super().__init__()
        self.smm = SplitAndMergeModule(in_G, n_band, n_split)
        self.downsample = nn.Conv2d(
            in_G, out_G, 
            kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

    def forward(self, x):
        skip_connection_data = self.smm(x)
        x = self.downsample(skip_connection_data)
        return x, skip_connection_data

class LightRoformerEncoder(nn.Module):
    """Encoder structure for hierarchical feature extraction."""
    def __init__(self, C=2, N_band=4, G=8, N_splitEnc=3):
        super().__init__()
        self.initial_stage = LightRoformerInitialStage(C, N_band, G)
        current_G = G * N_band 
        
        self.enc1 = EncoderBlock(N_band, N_splitEnc, current_G, current_G * 2)
        self.enc2 = EncoderBlock(N_band, N_splitEnc, current_G * 2, current_G * 3)
        self.enc3 = EncoderBlock(N_band, N_splitEnc, current_G * 3, current_G * 4)

    def forward(self, x):
        x = self.initial_stage(x)
        x, skip1 = self.enc1(x)
        x, _ = self.enc2(x)
        x, _ = self.enc3(x)
        return x, skip1

class Bottleneck(nn.Module):
    """Bottleneck module that captures global context at the deepest part of the model."""
    def __init__(self, channels, n_band, n_split=3, num_layers=5, num_heads=8, dropout=0.1):
        super().__init__()
        self.pre_split = SplitAndMergeModule(channels, n_band, n_split)
        self.layers = nn.ModuleList([
            InterleavedRoPEBlock(channels, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.pre_split(x)
        for layer in self.layers:
            x = layer(x)
        return x

class LightRoformerAsymmetricDecoder(nn.Module):
    """Asymmetric decoder that restores by combining encoder and bottleneck features."""
    def __init__(self, n_band, n_split, G):
        super().__init__()
        self.GN = G * n_band
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.GN * 4,
                out_channels=self.GN,
                kernel_size=(1, 8), 
                stride=(1, 8)
            ),
            nn.GroupNorm(16, self.GN),
            nn.ReLU()
        )        
        self.post_smm = SplitAndMergeModule(self.GN, n_band, n_split=1)

    def forward(self, x_bottleneck, x_skip_enc1):
        x_up = self.upsample(x_bottleneck)
        if x_up.shape[-1] != x_skip_enc1.shape[-1]:
            x_up = F.interpolate(x_up, size=x_skip_enc1.shape[2:], mode='bilinear', align_corners=False)
        
        # Gating-based combination
        x_combined = x_up * x_skip_enc1
        out = self.post_smm(x_combined)
        return out
