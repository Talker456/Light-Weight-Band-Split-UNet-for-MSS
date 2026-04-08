import torch
import torch.nn as nn
import torch.nn.functional as F

class LightRoformerInitialStage(nn.Module):
    """Module that splits audio input into subbands and extracts initial features."""
    def __init__(self, C=2, N_band=4, G=32):
        super().__init__()
        self.C = C           # Input channels (e.g., stereo 2 channels)
        self.N_band = N_band # Number of subbands (4)
        self.G = G           # Number of feature maps (channels) per band
        
        # Split Module (K=1): Group Convolution
        self.split_module_k1 = nn.Conv2d(
            in_channels=self.C * self.N_band, 
            out_channels=self.G * self.N_band, 
            kernel_size=1, 
            groups=self.N_band
        )

    def forward(self, x):
        # 1. Convert complex input to real channels
        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=1)
            
        B, C, F, T = x.shape
        
        # 2. Band Splitting
        x = x.view(B, C, self.N_band, F // self.N_band, T)
        
        # 3. Rearrange dimensions (Combine channels and bands)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, self.C * self.N_band, F // self.N_band, T)
        
        # 4. Pass through Split Module (K=1)
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
