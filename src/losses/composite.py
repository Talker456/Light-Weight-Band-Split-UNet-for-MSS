import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.stft import MultiResolutionSTFTLoss

class CompositeLoss(nn.Module):
    """A composite loss function combining waveform L1 loss and auraloss-based MR-STFT loss."""
    def __init__(self, sample_rate=44100):
        super(CompositeLoss, self).__init__()
        self.mr_stft_loss = MultiResolutionSTFTLoss(sample_rate)

    def forward(self, estimates, targets, audio_engine):
        total_loss = 0.0
        num_stems = len(estimates)
        
        for stem in estimates.keys():
            est_spec = estimates[stem]
            tgt_spec = targets[stem]
            
            # Convert spectrogram to waveform (istft)
            est_audio = audio_engine.istft(est_spec)
            tgt_audio = audio_engine.istft(tgt_spec)
            
            # Match length
            min_len = min(est_audio.shape[-1], tgt_audio.shape[-1])
            est_audio = est_audio[..., :min_len]
            tgt_audio = tgt_audio[..., :min_len]
            
            # 1. Waveform L1 Loss
            waveform_l1 = F.l1_loss(est_audio, tgt_audio)
            
            # 2. Activate MR-STFT Loss (auraloss-based)
            stft_loss = self.mr_stft_loss(est_audio, tgt_audio)
            
            # Sum for balancing the two losses (generally 1:1 ratio is used)
            total_loss += (waveform_l1 + stft_loss)
            
        return total_loss / num_stems
