import os
import torch
import torchaudio
import yaml
import argparse
import time
import sys
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.roformer.model import LightRoformer
from src.utils.audio import AudioEngine

def load_model(stem, config, device):
    model = LightRoformer(
        in_channels=2, out_channels=2,
        n_band=config['model'].get('num_bands', 4),
        G=config['model'].get('G', 8),
        n_layers=config['model'].get('n_rope', 5),
        n_heads=config['model'].get('num_heads', 8)
    ).to(device)
    
    model_path = os.path.join("checkpoints", stem, f"best_model_{stem}.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"✅ Loaded [{stem}]")
    else:
        print(f"⚠️ Warning: [{stem}] not found.")
        return None
    model.eval()
    return model

def separate(input_path, output_dir, config_path, target_stems=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = config['audio']['sample_rate']

    metadata = torchaudio.info(input_path)
    audio, sr = torchaudio.load(input_path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    if audio.shape[0] == 1: audio = audio.repeat(2, 1)
    
    chunk_samples = int(6.0 * sample_rate)
    overlap_samples = int(1.0 * sample_rate)
    hop_samples = chunk_samples - overlap_samples

    engine = AudioEngine(
        sample_rate=sample_rate,
        n_fft=config['audio'].get('n_fft', 6144),
        hop_length=config['audio'].get('hop_length', 1024),
        win_length=config['audio'].get('win_length', 6144)
    )
    
    stems_to_process = ["vocals", "bass", "drums", "other"] if target_stems is None or target_stems[0] == "all" else target_stems
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path).rsplit('.', 1)[0]
    
    window = torch.hann_window(overlap_samples * 2)
    chunk_weight = torch.ones(chunk_samples)
    chunk_weight[:overlap_samples] = window[:overlap_samples]
    chunk_weight[-overlap_samples:] = window[overlap_samples:]

    for stem in stems_to_process:
        model = load_model(stem, config, device)
        if model is None: continue
        
        total_samples = audio.shape[1]
        out_audio = torch.zeros((2, total_samples))
        weight_mask = torch.zeros((1, total_samples))
        
        for start in tqdm(range(0, total_samples, hop_samples), desc=f"Separating {stem}"):
            end = min(start + chunk_samples, total_samples)
            actual_len = end - start
            if actual_len < engine.hop_length: break
            
            chunk = audio[:, start:end]
            if actual_len < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))
            
            with torch.no_grad():
                spec = engine.stft(chunk.unsqueeze(0).to(device))
                est_spec = model(spec)
                est_chunk = engine.istft(est_spec, length=chunk_samples).squeeze(0).cpu()
            
            w = chunk_weight.clone()
            if start == 0: w[:overlap_samples] = 1.0
            if end == total_samples: w[-(total_samples - start):] = 1.0
            
            out_audio[:, start:end] += est_chunk[:, :actual_len] * w[:actual_len]
            weight_mask[:, start:end] += w[:actual_len]

        out_audio /= (weight_mask + 1e-10)
        save_path = os.path.join(output_dir, f"{filename}_{stem}.wav")
        
        # Clamp to -1.0 to 1.0 to prevent overflow
        out_audio = torch.clamp(out_audio, -1.0, 1.0)
        
        # Maintain original bit depth and encoding (prevent size explosion)
        # Saving as 16-bit PCM reduces file size by half compared to 32-bit float
        torchaudio.save(
            save_path, 
            out_audio, 
            sample_rate, 
            encoding="PCM_S", 
            bits_per_sample=16
        )
        print(f"✨ Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--stems", type=str, nargs="+", default=["all"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    separate(args.input, args.output, args.config, args.stems)
