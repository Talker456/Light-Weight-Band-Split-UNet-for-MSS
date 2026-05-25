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
    Unified dataset class supporting MUSDB18-HQ (WAV or FLAC) files and on-the-fly mixing.
    """
    def __init__(self, root_dir, sample_rate=44100, duration=4.0, is_train=True, samples_per_track=1, tracks=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * duration)
        self.is_train = is_train
        self.samples_per_track = samples_per_track
        self.audio_engine = AudioEngine(sample_rate=sample_rate)

        # Detection of file extension
        self.extension = self._detect_extension()

        if self.is_train:
            self.augment = Compose([
                ChannelShuffle(p=0.5),
                RandomGain(0.8, 1.1)
            ])
        else:
            self.augment = None

        if tracks is not None:
            self.tracks = tracks
        else:
            self.tracks = self._get_tracks()
            
        # Cache track lengths to avoid redundant metadata reads
        self.track_lengths = {}
        if self.is_train:
            print(f"Caching lengths for {len(self.tracks)} tracks ({self.extension})...")
            for track in self.tracks:
                stem_path = os.path.join(self.root_dir, track, f"vocals{self.extension}")
                if os.path.exists(stem_path):
                    try:
                        info = torchaudio.info(stem_path)
                        self.track_lengths[track] = info.num_frames
                    except:
                        self.track_lengths[track] = self.segment_length * 100

    def _detect_extension(self):
        """Detect whether the dataset uses .wav or .flac."""
        if not os.path.exists(self.root_dir):
            return ".flac"
            
        for root, dirs, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(".wav"): return ".wav"
                if f.endswith(".flac"): return ".flac"
        return ".flac"

    def _get_tracks(self):
        if not os.path.exists(self.root_dir):
            return []
            
        tracks = []
        target_file = f"vocals{self.extension}"
        
        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(full_path):
                if os.path.exists(os.path.join(full_path, target_file)):
                    tracks.append(entry)
                else:
                    # Search one level deeper for some dataset structures
                    for subentry in os.listdir(full_path):
                        sub_full_path = os.path.join(full_path, subentry)
                        if os.path.isdir(sub_full_path) and os.path.exists(os.path.join(sub_full_path, target_file)):
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
            if self.is_train:
                selected_idx = random.randint(0, len(self.tracks) - 1)
                track_name = self.tracks[selected_idx]
            else:
                track_name = self.tracks[idx]
                
            track_path = os.path.join(self.root_dir, track_name)
            stem_path = os.path.join(track_path, f"{stem}{self.extension}")
            
            if not os.path.exists(stem_path):
                stem_audio[stem] = torch.zeros(2, self.segment_length)
                continue

            if self.is_train:
                total_frames = self.track_lengths.get(track_name, self.segment_length * 10)
                start = 0
                if total_frames > self.segment_length:
                    start = random.randint(0, total_frames - self.segment_length)
                
                audio = self._load_audio(stem_path, offset=start, num_frames=self.segment_length)
                if audio.shape[1] < self.segment_length:
                    audio = F.pad(audio, (0, self.segment_length - audio.shape[1]))
            else:
                audio = self._load_audio(stem_path)
            
            if self.augment:
                audio = self.augment(audio)
            stem_audio[stem] = audio

        mixture = sum(stem_audio.values())
        mixture_spec = self.audio_engine.stft(mixture.unsqueeze(0)).squeeze(0)
        targets = {s: self.audio_engine.stft(a.unsqueeze(0)).squeeze(0) for s, a in stem_audio.items()}
        
        return mixture_spec, targets
