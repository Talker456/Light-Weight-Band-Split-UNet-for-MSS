import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.roformer.blocks import (
    LightRoformerEncoder, 
    Bottleneck, 
    LightRoformerAsymmetricDecoder
)

class LightRoformerFinalStage(nn.Module):
    """Layer that merges separated subband features back into a single full-band spectrogram."""
    def __init__(self, C_out, N_band, G):
        super().__init__()
        self.C_out = C_out
        self.N_band = N_band
        self.G = G
        self.merge_conv = nn.Conv2d(G * N_band, C_out * N_band, kernel_size=1, groups=N_band)

    def forward(self, x):
        x = self.merge_conv(x)
        B, _, F_sub, T = x.shape
        x = x.view(B, self.N_band, self.C_out, F_sub, T)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.C_out, self.N_band * F_sub, T)
        return x

class LightRoformer(nn.Module):
    """Overall Moises-Light model pipeline based on Interleaved RoPE Transformer."""
    def __init__(self, in_channels=2, out_channels=2, n_band=4, G=32, n_layers=5, n_heads=8):
        super().__init__()
        self.n_band = n_band
        self.G = G
        self.out_channels = out_channels
        
        # 1. Encoder (processes complex input by converting to real channels)
        self.encoder = LightRoformerEncoder(C=in_channels * 2, N_band=n_band, G=G, N_splitEnc=3)
        
        # 2. Bottleneck (Interleaved RoPE Transformer)
        bottleneck_channels = G * n_band * 4
        self.bottleneck = Bottleneck(
            channels=bottleneck_channels, 
            n_band=n_band, 
            n_split=3, 
            num_layers=n_layers, 
            num_heads=n_heads
        )
        
        # 3. Asymmetric decoder
        self.decoder = LightRoformerAsymmetricDecoder(n_band=n_band, n_split=1, G=G)
        
        # 4. Final output stage (band merging)
        self.final_stage = LightRoformerFinalStage(C_out=out_channels * 2, N_band=n_band, G=G)

    def forward(self, x):
        """
        Input x: (B, C, F, T) - Complex spectrogram
        Output: (B, C, F, T) - Estimated source complex spectrogram
        """
        with torch.amp.autocast(device_type='cuda', enabled=False):
            B, C, F_orig, T_orig = x.shape
            
            # 1. Padding: Align to multiples of N_band and stride (8x)
            pad_f = (self.n_band - (F_orig % self.n_band)) % self.n_band
            pad_t = (8 - (T_orig % 8)) % 8
            if pad_f > 0 or pad_t > 0:
                x = F.pad(x, (0, pad_t, 0, pad_f))
        
        # A. Encoding and feature compression
        latent, skip1 = self.encoder(x)
        
        # B. Interleaved time-frequency modeling
        latent = self.bottleneck(latent)
        
        # C. Decoding (Gating-based combination)
        decoded = self.decoder(latent, skip1)
        
        # D. Final output restoration (real channel format)
        out = self.final_stage(decoded)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # E. Remove padding
            if pad_f > 0 or pad_t > 0:
                out = out[:, :, :F_orig, :T_orig]
            
            # F. Convert back to complex tensor
            real, imag = out.chunk(2, dim=1)
            out_complex = torch.complex(real.float(), imag.float())
        
        return out_complex
