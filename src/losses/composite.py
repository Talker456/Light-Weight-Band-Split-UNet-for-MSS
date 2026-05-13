import torch
import torch.nn as nn
import torch.nn.functional as F

def spec_rmse_loss(spec_estimate, spec_sources):
    """
    Calculate Spectral RMSE loss directly from complex spectrograms.
    1. Ensure frame alignment
    2. Convert complex to real (stacking real/imag)
    3. Calculate MSE then RMSE
    """
    # Match frames (last dimension) just in case
    min_frames = min(spec_estimate.shape[-1], spec_sources.shape[-1])
    spec_estimate = spec_estimate[..., :min_frames]
    spec_sources = spec_sources[..., :min_frames]

    # 1. View as real (adds a dimension of size 2 at the end: [Real, Imag])
    # Expects input shape (Batch, Channels, Freqs, Frames) complex
    # Output shape (Batch, Channels, Freqs, Frames, 2)
    spec_estimate = torch.view_as_real(spec_estimate)
    spec_sources = torch.view_as_real(spec_sources)

    # 2. MSE Loss (none reduction to handle mean/sqrt manually)
    loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')

    # 3. RMSE calculation
    # Mean over Freqs, Frames, and Real/Imag dimensions (dims 2, 3, 4)
    dims = tuple(range(2, loss.dim()))
    loss = loss.mean(dims).sqrt().mean(dim=(0, 1))  # Mean over Batch and Channels

    return loss

class CompositeLoss(nn.Module):
    """A loss function using Spectral RMSE loss calculated directly on spectrograms."""
    def __init__(self, sample_rate=44100):
        super(CompositeLoss, self).__init__()
        self.sample_rate = sample_rate

    def forward(self, estimates, targets, audio_engine=None):
        """
        estimates: Dict of {stem: spectrogram_tensor}
        targets: Dict of {stem: spectrogram_tensor}
        audio_engine: Not needed for direct spec loss, but kept for signature compatibility
        """
        total_loss = 0.0
        num_stems = len(estimates)
        
        for stem in estimates.keys():
            est_spec = estimates[stem]
            tgt_spec = targets[stem]
            
            # Directly calculate Spectral RMSE Loss on the spectrograms
            # This avoids redundant ISTFT -> STFT operations
            spectral_rmse = spec_rmse_loss(est_spec, tgt_spec)
            
            total_loss += spectral_rmse
            
        return total_loss / num_stems
