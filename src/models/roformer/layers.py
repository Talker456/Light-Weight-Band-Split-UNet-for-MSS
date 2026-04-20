import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_band_indices(num_freq_bins, n_band):
    """
    Computes log-spaced band indices for non-uniform frequency splitting.
    Returns a list of (start_idx, end_idx) for each band.
    """
    # Using log-space to give more resolution to lower frequencies
    # We use indices from 0 to num_freq_bins
    # np.geomspace or np.logspace can be used. 
    # Let's use a simple power-law or log-space.
    indices = np.unique(np.logspace(0, np.log10(num_freq_bins), n_band + 1).astype(int))
    
    # If unique reduces the number of bands, we fallback to linear for the remaining
    if len(indices) < n_band + 1:
        indices = np.linspace(0, num_freq_bins, n_band + 1).astype(int)
    
    # Ensure the last index is exactly num_freq_bins
    indices[-1] = num_freq_bins
    indices[0] = 0
    
    bands = []
    for i in range(len(indices) - 1):
        bands.append((indices[i], indices[i+1]))
    
    # If we have fewer bands than requested due to small num_freq_bins, 
    # it's handled by the length of returned bands.
    return bands

class MelBandSplit(nn.Module):
    """Initial stage using non-uniform (log-spaced) banding."""
    def __init__(self, in_channels=2, num_freq_bins=1025, n_band=12, G=32):
        super().__init__()
        self.in_channels = in_channels # e.g. 4 (2 real + 2 imag for stereo)
        self.n_band = n_band
        self.G = G
        self.band_indices = get_band_indices(num_freq_bins, n_band)
        self.n_band = len(self.band_indices) # Update n_band in case of duplicates
        
        self.projections = nn.ModuleList([
            nn.Linear((end - start) * in_channels, G)
            for start, end in self.band_indices
        ])

    def forward(self, x):
        # Input x: (B, C_in, F, T) - Real format
        B, C, F, T = x.shape
        
        band_features = []
        for i, (start, end) in enumerate(self.band_indices):
            # Extract band: (B, C, F_sub, T)
            subband = x[:, :, start:end, :]
            # Flatten C and F_sub: (B, T, C * F_sub)
            subband = subband.permute(0, 3, 1, 2).reshape(B, T, -1)
            # Project: (B, T, G)
            proj = self.projections[i](subband)
            band_features.append(proj)
        
        # Stack bands: (B, T, N_band, G)
        out = torch.stack(band_features, dim=2)
        # Permute to (B, G * N_band, 1, T) for compatibility with existing modules
        # or (B, G*N_band, F_dummy, T) where F_dummy=1
        out = out.permute(0, 3, 2, 1).reshape(B, self.G * self.n_band, 1, T)
        return out

class MelBandMerge(nn.Module):
    """Final stage to merge features back into non-uniform bands."""
    def __init__(self, G=32, n_band=12, out_channels=2, num_freq_bins=1025):
        super().__init__()
        self.G = G
        self.n_band = n_band
        self.out_channels = out_channels
        self.band_indices = get_band_indices(num_freq_bins, n_band)
        
        self.projections = nn.ModuleList([
            nn.Linear(G, (end - start) * out_channels)
            for start, end in self.band_indices
        ])

    def forward(self, x):
        # Input x: (B, G * N_band, 1, T)
        B, GN, _, T = x.shape
        x = x.view(B, self.G, self.n_band, T).permute(0, 3, 2, 1) # (B, T, N_band, G)
        
        outs = []
        for i, (start, end) in enumerate(self.band_indices):
            # Project: (B, T, F_sub * C_out)
            proj = self.projections[i](x[:, :, i, :])
            # Reshape: (B, T, C_out, F_sub) -> (B, C_out, F_sub, T)
            proj = proj.view(B, T, self.out_channels, end - start).permute(0, 2, 3, 1)
            outs.append(proj)
            
        # Concatenate bands along frequency: (B, C_out, F, T)
        out = torch.cat(outs, dim=2)
        return out

class LightRoformerInitialStage(nn.Module):
    """
    Legacy wrapper for LightRoformerInitialStage. 
    Now uses MelBandSplit internally if non-uniform is desired, 
    but for now let's keep it compatible or replace it.
    """
    def __init__(self, C=2, N_band=4, G=32, num_freq_bins=None):
        super().__init__()
        if num_freq_bins is not None:
            self.impl = MelBandSplit(in_channels=C, num_freq_bins=num_freq_bins, n_band=N_band, G=G)
        else:
            # Fallback to uniform if num_freq_bins is not provided
            self.C = C
            self.N_band = N_band
            self.G = G
            self.split_module_k1 = nn.Conv2d(
                in_channels=self.C * self.N_band, 
                out_channels=self.G * self.N_band, 
                kernel_size=1, 
                groups=self.N_band
            )
            self.impl = None

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=1)
            
        if self.impl is not None:
            return self.impl(x)
            
        B, C, F, T = x.shape
        x = x.view(B, C, self.N_band, F // self.N_band, T)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, self.C * self.N_band, F // self.N_band, T)
        out = self.split_module_k1(x)
        return out

class SplitModuleK3(nn.Module):
    """Split Module with K=3: Group Convolution that processes each subband independently."""
    def __init__(self, channels, n_band):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 
            kernel_size=3, padding=1, 
            groups=n_band
        )
        
    def forward(self, x):
        return self.conv(x)

class SplitAndMergeModule(nn.Module):
    """Module that performs independent subband operations (Split) and cross-channel operations (Merge)."""
    def __init__(self, channels, n_band, n_split):
        super().__init__()
        self.split_repeater_1 = nn.Sequential(
            *[SplitModuleK3(channels, n_band) for _ in range(n_split)]
        )
        self.split_repeater_2 = nn.Sequential(
            *[SplitModuleK3(channels, n_band) for _ in range(n_split)]
        )
        
        self.fc_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        self.residual_3x3_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Top path (Split + FC + Split)
        x_top = self.split_repeater_1(x)
        x_fc = self.fc_path(x_top)
        x_top = x_fc + x_top 
        x_top = self.split_repeater_2(x_top)
        
        # Bottom path (Residual Path)
        x_bottom = self.residual_3x3_conv(x)
        
        return x_top + x_bottom
