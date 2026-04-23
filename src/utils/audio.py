import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEngine:
    def __init__(self, n_fft=6144, hop_length=1024, win_length=6144, sample_rate=44100):
        """
        Initialize the Audio Engine with paper-specific STFT parameters.
        Paper: Window 6144, Hop 1024.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # Hann window is commonly used for audio
        self.register_window()

    def register_window(self):
        self.window = torch.hann_window(self.win_length)

    def stft(self, y):
        """
        Compute STFT of the input signal.
        Input y: (Batch, Channels, Samples)
        Output: (Batch, Channels, Freqs, Frames, Real/Imag)
        """
        # Ensure input and window are float32 for numerical stability
        y = y.float()
        window = self.window.to(y.device).float()
        
        batch_size, channels, samples = y.shape
        y = y.view(batch_size * channels, samples)
        
        spec = torch.stft(
            y, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=window, 
            center=True, 
            pad_mode='reflect', 
            normalized=False, 
            onesided=True,
            return_complex=True
        )
        
        # Reshape back to (Batch, Channels, Freqs, Frames)
        _, freqs, frames = spec.shape
        spec = spec.view(batch_size, channels, freqs, frames)
        
        return spec

    def istft(self, spec, length=None):
        """
        Compute inverse STFT.
        Input spec: (Batch, Channels, Freqs, Frames) complex tensor
        Output: (Batch, Channels, Samples)
        """
        # Ensure spec and window are complex64/float32 for precision
        if torch.is_complex(spec):
            spec = spec.to(torch.complex64)
        else:
            spec = spec.float()
            
        window = self.window.to(spec.device).float()
        batch_size, channels, freqs, frames = spec.shape
        
        spec = spec.view(batch_size * channels, freqs, frames)
        
        y = torch.istft(
            spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=window, 
            center=True, 
            normalized=False, 
            onesided=True,
            length=length
        )
        
        y = y.view(batch_size, channels, -1)
        return y
