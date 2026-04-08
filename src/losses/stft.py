import torch
import torch.nn as nn
import auraloss

class MultiResolutionSTFTLoss(nn.Module):
    """High-performance multi-resolution STFT loss function using the auraloss library."""
    def __init__(self, sample_rate=44100, device="cpu"):
        super(MultiResolutionSTFTLoss, self).__init__()
        
        # Maintain the resolution settings used in the existing project
        self.fft_sizes = [6144, 3072, 1536, 768, 384]
        self.hop_sizes = [1536, 768, 384, 192, 96]
        self.win_lengths = [6144, 3072, 1536, 768, 384]
        
        # Initialize MultiResolutionSTFTLoss from auraloss
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=self.fft_sizes,
            hop_sizes=self.hop_sizes,
            win_lengths=self.win_lengths,
            scale="mel",
            n_bins=64,
            sample_rate=sample_rate,
            perceptual_weighting=True
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: Predicted audio waveform (B, C, S)
        y: Target audio waveform (B, C, S)
        """
        # auraloss natively supports (B, C, S) format.
        return self.mrstft(x, y)
