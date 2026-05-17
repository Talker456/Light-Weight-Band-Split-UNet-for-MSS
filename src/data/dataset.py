import os
import random
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from src.utils.audio import AudioEngine
from src.data.augmentations import Compose, ChannelShuffle, GaussianNoise, RandomGain

class MUSDBDataset(Dataset):
    """
    Unified dataset class supporting MUSDB18-HQ FLAC files and on-the-fly mixing.
    """
    def __init__(self, root_dir, sample_rate=44100, duration=4.0, is_train=True, samples_per_track=1, tracks=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * duration)
        self.is_train = is_train
        self.samples_per_track = samples_per_track
        self.audio_engine = AudioEngine(sample_rate=sample_rate)

        if self.is_train:
            self.augment = Compose([
                ChannelShuffle(p=0.5),
                RandomGain(0.8, 1.1)
                # GaussianNoise(std=0.005, p=0.2)
            ])
        else:
            self.augment = None

        if tracks is not None:
            self.tracks = tracks
        else:
            self.tracks = self._get_tracks()

    def _get_tracks(self):
        if not os.path.exists(self.root_dir):
            return []
            
        tracks = []
        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(full_path):
                if os.path.exists(os.path.join(full_path, 'vocals.flac')):
                    tracks.append(entry)
                else:
                    for subentry in os.listdir(full_path):
                        sub_full_path = os.path.join(full_path, subentry)
                        if os.path.isdir(sub_full_path) and os.path.exists(os.path.join(sub_full_path, 'vocals.flac')):
                            tracks.append(os.path.join(entry, subentry))
        return tracks

    def __len__(self):
        if self.is_train:
            return len(self.tracks) * self.samples_per_track
        return len(self.tracks)

    def _load_audio(self, path, offset=None, num_frames=None):
        if offset is not None and num_frames is not None:
            audio, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
        else:
            audio, sr = torchaudio.load(path)
            
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        return audio

    def __getitem__(self, idx):
        stems = ['vocals', 'drums', 'bass', 'other']
        stem_audio = {}
        
        for stem in stems:
            # In training mode, select each stem randomly from different tracks (Inter-track Mixing)
            if self.is_train:
                selected_idx = random.randint(0, len(self.tracks) - 1)
            else:
                selected_idx = idx
                
            track_path = os.path.join(self.root_dir, self.tracks[selected_idx])
            stem_path = os.path.join(track_path, f"{stem}.flac")
            
            # Handling if the corresponding stem file does not exist
            if not os.path.exists(stem_path):
                # Since the stem might also be missing in other songs, search iteratively or fill with zeros
                stem_audio[stem] = torch.zeros(2, self.segment_length)
                continue

            # Determine loading parameters based on mode
            if self.is_train:
                try:
                    # Determine random crop point independently according to the length of each song
                    info = torchaudio.info(stem_path)
                    total_frames = info.num_frames
                except:
                    audio_tmp, _ = torchaudio.load(stem_path)
                    total_frames = audio_tmp.shape[1]
                    del audio_tmp

                start = 0
                if total_frames > self.segment_length:
                    start = random.randint(0, total_frames - self.segment_length)
                
                # Audio loading and preprocessing
                audio = self._load_audio(stem_path, offset=start, num_frames=self.segment_length)
                
                # Padding if length is insufficient
                if audio.shape[1] < self.segment_length:
                    audio = F.pad(audio, (0, self.segment_length - audio.shape[1]))
            else:
                # Load full audio for validation
                audio = self._load_audio(stem_path)
            
            # Apply data augmentation
            if self.augment:
                audio = self.augment(audio)
                
            stem_audio[stem] = audio

        # Combine stems extracted from different songs to create a new mixture
        mixture = sum(stem_audio.values())
        
        # Spectrogram transformation
        mixture_spec = self.audio_engine.stft(mixture.unsqueeze(0)).squeeze(0)
        targets = {s: self.audio_engine.stft(a.unsqueeze(0)).squeeze(0) for s, a in stem_audio.items()}
        
        return mixture_spec, targets
