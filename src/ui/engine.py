import os
import yaml
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import museval
from pedalboard import Pedalboard, Gain, LowShelfFilter, HighShelfFilter, PeakFilter
from src.models.roformer.model import LightRoformer
from src.utils.audio import AudioEngine

class AudioSeparatorEngine:
    def __init__(self, config_path="configs/default.yaml", checkpoint_dir="checkpoints"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = self.config['audio']['sample_rate']
        self.checkpoint_dir = checkpoint_dir
        
        self.audio_engine = AudioEngine(
            n_fft=self.config['audio'].get('n_fft', 6144),
            hop_length=self.config['audio'].get('hop_length', 1024),
            win_length=self.config['audio'].get('win_length', 6144),
            sample_rate=self.sample_rate
        )
        
        self.stems = ["vocals", "bass", "drums", "other"]
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all 4 stem models into memory."""
        for stem in self.stems:
            model = LightRoformer(
                in_channels=2, out_channels=2,
                n_band=self.config['model'].get('num_bands', 4),
                G=self.config['model'].get('G', 8),
                n_layers=self.config['model'].get('n_rope', 5),
                n_heads=self.config['model'].get('num_heads', 8)
            ).to(self.device)
            
            model_path = os.path.join(self.checkpoint_dir, stem, f"best_model_{stem}.pth")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict)
                model.eval()
                self.models[stem] = model
            else:
                print(f"Warning: Checkpoint for {stem} not found at {model_path}")
                self.models[stem] = None

    def apply_effects(self, audio_np, eq_params):
        """
        Apply EQ and Gain using Pedalboard.
        audio_np: (Channels, Samples) numpy array
        eq_params: dict with keys 'low', 'mid', 'high', 'gain' (all in dB)
        """
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=200, gain_db=eq_params.get('low', 0)),
            PeakFilter(cutoff_frequency_hz=1000, gain_db=eq_params.get('mid', 0)),
            HighShelfFilter(cutoff_frequency_hz=5000, gain_db=eq_params.get('high', 0)),
            Gain(gain_db=eq_params.get('gain', 0))
        ])
        # Pedalboard expects (Samples, Channels) or (Channels, Samples)
        return board(audio_np, self.sample_rate)

    def separate_audio(self, mix_audio, target_stems=None, callback=None):
        """
        Separate mixture audio into stems.
        mix_audio: (2, Samples) torch tensor
        """
        if target_stems is None:
            target_stems = self.stems
            
        total_samples = mix_audio.shape[1]
        chunk_samples = int(6.0 * self.sample_rate)
        overlap_samples = int(1.0 * self.sample_rate)
        hop_samples = chunk_samples - overlap_samples
        
        # Cross-fading window
        window = torch.hann_window(overlap_samples * 2)
        chunk_weight = torch.ones(chunk_samples)
        chunk_weight[:overlap_samples] = window[:overlap_samples]
        chunk_weight[-overlap_samples:] = window[overlap_samples:]
        
        results = {}
        for stem in target_stems:
            if self.models[stem] is None: continue
            
            out_audio = torch.zeros((2, total_samples))
            weight_mask = torch.zeros((1, total_samples))
            
            steps = range(0, total_samples, hop_samples)
            for i, start in enumerate(steps):
                if callback:
                    callback(stem, int((i / len(steps)) * 100))
                    
                end = min(start + chunk_samples, total_samples)
                actual_len = end - start
                if actual_len < self.audio_engine.hop_length: break
                
                chunk = mix_audio[:, start:end]
                if actual_len < chunk_samples:
                    chunk = F.pad(chunk, (0, chunk_samples - actual_len))
                
                with torch.no_grad():
                    spec = self.audio_engine.stft(chunk.unsqueeze(0).to(self.device))
                    est_spec = self.models[stem](spec)
                    est_chunk = self.audio_engine.istft(est_spec, length=chunk_samples).squeeze(0).cpu()
                
                w = chunk_weight.clone()
                if start == 0: w[:overlap_samples] = 1.0
                if end == total_samples: w[-(total_samples - start):] = 1.0
                
                out_audio[:, start:end] += est_chunk[:, :actual_len] * w[:actual_len]
                weight_mask[:, start:end] += w[:actual_len]
                
            out_audio /= (weight_mask + 1e-10)
            results[stem] = torch.clamp(out_audio, -1.0, 1.0)
            
        return results

    def get_spectrogram(self, audio_tensor):
        """Returns dB-spectrogram normalized relative to max (0dB)."""
        with torch.no_grad():
            spec = self.audio_engine.stft(audio_tensor.unsqueeze(0).to(self.device))
            mag = torch.abs(spec).mean(dim=1).squeeze(0).cpu().numpy()
            
            # Normalization: 20 * log10(mag / max(mag))
            mag_max = np.max(mag)
            db_spec = 20 * np.log10(mag / (mag_max + 1e-10) + 1e-10)
            
        return db_spec

    def calculate_bss_eval(self, reference_dict, estimate_dict):
        """Calculate SDR, SIR, SAR using museval."""
        stems = list(reference_dict.keys())
        # Prepare arrays (Stems, Samples, Channels)
        ref = np.stack([reference_dict[s].numpy().T for s in stems], axis=0)
        est = np.stack([estimate_dict[s].numpy().T for s in stems], axis=0)
        
        # museval expects float64
        scores = museval.evaluate(ref.astype(np.float64), est.astype(np.float64))
        
        results = {}
        for i, stem in enumerate(stems):
            results[stem] = {
                'SDR': np.nanmedian(scores[0][i]),
                'SIR': np.nanmedian(scores[1][i]),
                'SAR': np.nanmedian(scores[2][i])
            }
        return results

    def save_audio(self, audio_tensor, save_path):
        """Save torch tensor to WAV file (16-bit PCM)."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Clamp to -1.0 to 1.0 to prevent overflow
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        torchaudio.save(
            save_path, 
            audio_tensor, 
            self.sample_rate, 
            encoding="PCM_S", 
            bits_per_sample=16
        )
        return save_path
