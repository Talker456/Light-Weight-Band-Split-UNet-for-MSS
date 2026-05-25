import torch
import random
import torchaudio.transforms as T
import torch.nn.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

class ChannelShuffle:
    """Randomly swaps the left and right channels of stereo audio."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, audio):
        if audio.shape[0] == 2 and random.random() < self.p:
            return audio.flip(0)
        return audio

class GaussianNoise:
    """Adds Gaussian noise."""
    def __init__(self, std=0.01, p=0.3):
        self.std = std
        self.p = p
    def __call__(self, audio):
        if random.random() < self.p:
            noise = torch.randn_like(audio) * self.std
            return audio + noise
        return audio

class RandomGain:
    """Randomly adjusts the audio amplitude."""
    def __init__(self, min_gain=0.7, max_gain=1.2):
        self.min_gain = min_gain
        self.max_gain = max_gain
    def __call__(self, audio):
        gain = random.uniform(self.min_gain, self.max_gain)
        return audio * gain

class PitchShift:
    """Randomly adjusts the pitch."""
    def __init__(self, sample_rate, n_steps=2, p=0.2):
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            steps = random.uniform(-self.n_steps, self.n_steps)
            if abs(steps) > 0.1:
                return T.PitchShift(self.sample_rate, n_steps=steps)(audio)
        return audio
