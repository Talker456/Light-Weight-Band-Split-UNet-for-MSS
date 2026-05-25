import os
import sys
import yaml
import torch
import torchaudio
import numpy as np
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.roformer.model import LightRoformer
from src.utils.audio import AudioEngine

class SegmentSeparator:
    def __init__(self, config_path, checkpoint_dir='checkpoints', device=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            print("WARNING: Using CPU")
            self.device = torch.device('cpu')
            
        self.checkpoint_dir = checkpoint_dir
        self.sample_rate = self.config['audio']['sample_rate']
        
        # Audio Engine for STFT/iSTFT
        self.engine = AudioEngine(
            sample_rate=self.sample_rate,
            n_fft=self.config['audio'].get('n_fft', 6144),
            hop_length=self.config['audio'].get('hop_length', 1024),
            win_length=self.config['audio'].get('win_length', 6144)
        )
        
        self.models = {}

    @property
    def instruments(self):
        return ['vocals', 'bass', 'drums', 'other']

    def load_model_for_stem(self, stem):
        if stem in self.models:
            return self.models[stem]
            
        model = LightRoformer(
            in_channels=2, out_channels=2,
            n_band=self.config['model'].get('num_bands', 4),
            G=self.config['model'].get('G', 8),
            n_layers=self.config['model'].get('n_rope', 5),
            n_heads=self.config['model'].get('num_heads', 8),
            bottleneck_type=self.config['model'].get('bottleneck_type', 'attention')
        ).to(self.device)
        
        # Check standard checkpoint paths
        model_path = os.path.join(self.checkpoint_dir, stem, f"best_model_{stem}.pth")
        
        # Fallback to secondary pattern if best_model_stem.pth is not found
        if not os.path.exists(model_path):
            stem_dir = os.path.join(self.checkpoint_dir, stem)
            if os.path.exists(stem_dir):
                files = [f for f in os.listdir(stem_dir) if f.startswith("best_model_") and f.endswith(".pth")]
                if files:
                    model_path = os.path.join(stem_dir, files[0])

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print(f"✅ Loaded [{stem}] checkpoint from {model_path}")
        else:
            print(f"⚠️ Warning: [{stem}] checkpoint not found at {model_path}. Using random weights.")
            
        model.eval()
        self.models[stem] = model
        return model

    def save_spectrogram(self, wav, sr, save_path, start_time, instrument_name):
        """
        Saves a professional spectrogram using librosa and matplotlib.
        """
        # Convert torch tensor to numpy [samples]
        if wav.shape[0] > 1:
            y = wav.mean(0).detach().cpu().numpy()
        else:
            y = wav.squeeze(0).detach().cpu().numpy()

        # Compute STFT and convert to dB
        S = np.abs(librosa.stft(y))
        D = librosa.amplitude_to_db(S, ref=np.max)

        # Visualization
        plt.figure(figsize=(14, 6))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')

        # Time axis correction (offset by start_time)
        tick_locs = plt.xticks()[0]
        plt.xticks(tick_locs, [f"{t + start_time:.1f}" for t in tick_locs])

        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f"Separated {instrument_name.capitalize()} Spectrogram ({start_time:.1f}s - {start_time + len(y)/sr:.1f}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        
        # Save and close
        plt.savefig(save_path)
        plt.close()

    def separate_segment(self, mixture_path, start_time, end_time, output_dir, target_stems=None):
        # 1. Get audio metadata & calculate segment frames
        metadata = torchaudio.info(mixture_path)
        orig_sr = metadata.sample_rate
        
        start_frame = int(start_time * orig_sr) if start_time is not None else 0
        if end_time is not None:
            frames = int((end_time - start_time) * orig_sr)
            if frames <= 0:
                print(f"Error: end_time ({end_time}) must be greater than start_time ({start_time})")
                return
        else:
            frames = -1
            
        print(f"Loading segment from {start_time if start_time else 0}s to {end_time if end_time else 'end'}s...")
        audio, sr = torchaudio.load(mixture_path, frame_offset=start_frame, num_frames=frames)
        
        # 2. Preprocess & Resample if necessary
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            sr = self.sample_rate
            
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
            
        # Z-score Normalization
        mono = audio.mean(dim=0)
        mean = mono.mean()
        std = mono.std()
        audio_norm = (audio - mean) / (std + 1e-10)
        
        # Overlap-add setup
        chunk_samples = int(6.0 * sr)
        overlap_samples = int(1.0 * sr)
        hop_samples = chunk_samples - overlap_samples
        
        window = torch.hann_window(overlap_samples * 2)
        chunk_weight = torch.ones(chunk_samples)
        chunk_weight[:overlap_samples] = window[:overlap_samples]
        chunk_weight[-overlap_samples:] = window[overlap_samples:]
        
        stems_to_process = self.instruments if target_stems is None or target_stems[0] == "all" else target_stems
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(mixture_path).rsplit('.', 1)[0]
        actual_start_time = start_time if start_time is not None else 0.0
        
        # 3. Inference
        for stem in stems_to_process:
            model = self.load_model_for_stem(stem)
            
            total_samples = audio_norm.shape[1]
            out_audio = torch.zeros((2, total_samples))
            weight_mask = torch.zeros((1, total_samples))
            
            for start in tqdm(range(0, total_samples, hop_samples), desc=f"Separating {stem}"):
                end = min(start + chunk_samples, total_samples)
                actual_len = end - start
                if actual_len < self.engine.hop_length: 
                    break
                
                chunk = audio_norm[:, start:end]
                if actual_len < chunk_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))
                
                with torch.no_grad():
                    spec = self.engine.stft(chunk.unsqueeze(0).to(self.device))
                    est_spec = model(spec)
                    est_chunk = self.engine.istft(est_spec, length=chunk_samples).squeeze(0).cpu()
                
                w = chunk_weight.clone()
                if start == 0: 
                    w[:overlap_samples] = 1.0
                if end == total_samples: 
                    w[-(total_samples - start):] = 1.0
                
                out_audio[:, start:end] += est_chunk[:, :actual_len] * w[:actual_len]
                weight_mask[:, start:end] += w[:actual_len]
                
            out_audio /= (weight_mask + 1e-10)
            
            # Inverse Normalization
            out_audio = out_audio * std + mean
            out_audio = torch.clamp(out_audio, -1.0, 1.0)
            
            # 4. Save results
            wav_path = os.path.join(output_dir, f"{filename}_{stem}.wav")
            torchaudio.save(
                wav_path, 
                out_audio, 
                sr, 
                encoding="PCM_S", 
                bits_per_sample=16
            )
            print(f"✨ Saved WAV: {wav_path}")
            
            # Save Spectrogram
            spec_path = os.path.join(output_dir, f"{filename}_{stem}_spec.png")
            self.save_spectrogram(out_audio, sr, spec_path, actual_start_time, stem)
            print(f"🎨 Saved Spectrogram: {spec_path}")

def main():
    parser = argparse.ArgumentParser(description="Segment-based Music Source Separation with Spectrograms")
    parser.add_argument('--mixture', type=str, required=True, help='Path to mixture audio file')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Path to directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='results_segment', help='Directory to save results')
    parser.add_argument('--stems', type=str, nargs="+", default=["all"], help='Stems to separate')
    
    args = parser.parse_args()
    
    separator = SegmentSeparator(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir
    )
    
    separator.separate_segment(
        mixture_path=args.mixture,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=args.output_dir,
        target_stems=args.stems
    )

if __name__ == "__main__":
    main()
