import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.roformer.blocks import (
    LightRoformerEncoder, 
    Bottleneck, 
    LightRoformerAsymmetricDecoder,
    GLUMaskEstimator
)
from src.models.roformer.layers import MelBandMerge

class LightRoformerFinalStage(nn.Module):
    """Layer that estimates a complex mask and merges subband features."""
    def __init__(self, C_out, N_band, G, num_freq_bins=3073):
        super().__init__()
        self.mask_estimator = GLUMaskEstimator(G, N_band)
        self.merge = MelBandMerge(G, N_band, C_out, num_freq_bins)

    def forward(self, x):
        # x: (B, G * N_band, 1, T)
        x = self.mask_estimator(x)
        mask = self.merge(x) # (B, C_out, F, T)
        return mask

class LightRoformer(nn.Module):
    """Overall Moises-Light model pipeline with Mel-banding and Complex Masking."""
    def __init__(self, in_channels=2, out_channels=2, n_band=4, G=32, n_layers=5, n_heads=8, num_freq_bins=3073):
        super().__init__()
        self.n_band = n_band
        self.G = G
        self.out_channels = out_channels
        self.num_freq_bins = num_freq_bins
        
        # 1. Encoder (processes complex input by converting to real channels)
        self.encoder = LightRoformerEncoder(
            C=in_channels * 2, 
            N_band=n_band, 
            G=G, 
            N_splitEnc=3, 
            num_freq_bins=num_freq_bins
        )
        
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
        
        # 4. Final output stage (Mask Estimation & Band Merging)
        self.final_stage = LightRoformerFinalStage(
            C_out=out_channels * 2, 
            N_band=n_band, 
            G=G, 
            num_freq_bins=num_freq_bins
        )

    def forward(self, x):
        """
        Input x: (B, C, F, T) - Complex spectrogram
        Output: (B, C, F, T) - Estimated source complex spectrogram
        """
        B, C, F_orig, T_orig = x.shape
        x_orig = x
        
        # 1. Padding: Align to multiples of stride (8x) 
        # F-padding is no longer strictly needed for MelBandSplit as it handles variable F,
        # but we might still want to pad to num_freq_bins or a multiple of something.
        # For MelBandSplit, it expects self.num_freq_bins.
        
        pad_f = (self.num_freq_bins - F_orig)
        pad_t = (8 - (T_orig % 8)) % 8
        
        if pad_f > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_f))
        
        # A. Encoding and feature compression
        latent, skip1 = self.encoder(x)
        
        # B. Interleaved time-frequency modeling
        latent = self.bottleneck(latent)
        
        # C. Decoding (Gating-based combination)
        decoded = self.decoder(latent, skip1)
        
        # D. Final mask estimation
        mask = self.final_stage(decoded) # (B, out_channels * 2, F_padded, T_padded)
        
        # E. Apply Mask (Complex Multiplication)
        mask_real, mask_imag = mask.chunk(2, dim=1)
        mask_complex = torch.complex(mask_real, mask_imag)
        
        # Masking: Y = X * M
        out_complex = x * mask_complex
        
        # F. Remove padding
        if pad_f > 0 or pad_t > 0:
            out_complex = out_complex[:, :, :F_orig, :T_orig]
        
        return out_complex
