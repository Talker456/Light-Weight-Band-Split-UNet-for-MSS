import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM

class DualPathRNN(nn.Module):
    """
    Dual-Path RNN module as used in SCNet, optimized for (B, C, F, T) tensors.
    Models inter-dependencies across Frequency and Time dimensions.
    """
    def __init__(self, channels, expand=1, bidirectional=True):
        super(DualPathRNN, self).__init__()

        self.channels = channels
        self.hidden_size = channels * expand
        self.bidirectional = bidirectional
        
        # LSTM layers for Frequency and Time paths
        # Frequency-path LSTM
        self.lstm_f = LSTM(
            input_size=channels, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        # Time-path LSTM
        self.lstm_t = LSTM(
            input_size=channels, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            bidirectional=bidirectional, 
            batch_first=True
        )

        # Projection layers to return to original channel size
        proj_dim = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.proj_f = nn.Linear(proj_dim, channels)
        self.proj_t = nn.Linear(proj_dim, channels)

        # Normalization layers
        self.norm_f = nn.GroupNorm(1, channels)
        self.norm_t = nn.GroupNorm(1, channels)

    def forward(self, x):
        """
        Input x: (B, C, F, T)
        """
        B, C, F, T = x.shape
        
        # 1. Frequency-path modeling
        res_f = x
        x = self.norm_f(x)
        # Reshape for LSTM: (Batch * Time, Sequence=Freq, Channels)
        x = x.permute(0, 3, 2, 1).contiguous().view(B * T, F, C)
        x, _ = self.lstm_f(x)
        x = self.proj_f(x)
        # Reshape back: (Batch, Channels, Freq, Time)
        x = x.view(B, T, F, C).permute(0, 3, 2, 1).contiguous()
        x = x + res_f

        # 2. Time-path modeling
        res_t = x
        x = self.norm_t(x)
        # Reshape for LSTM: (Batch * Freq, Sequence=Time, Channels)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)
        x, _ = self.lstm_t(x)
        x = self.proj_t(x)
        # Reshape back: (Batch, Channels, Freq, Time)
        x = x.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()
        x = x + res_t

        return x
