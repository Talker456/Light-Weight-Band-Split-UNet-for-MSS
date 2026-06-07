import os
import sys
import yaml
import torch
import torchaudio
import numpy as np
import argparse
import logging
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.roformer.model import LightRoformer
from src.utils.audio import AudioEngine, calculate_sdr

logger = logging.getLogger(__name__)

def load_model(stem, config, device):
    model = LightRoformer(
        in_channels=2, out_channels=2,
        n_band=config['model'].get('num_bands', 4),
        G=config['model'].get('G', 8),
        n_layers=config['model'].get('n_rope', 5),
        n_heads=config['model'].get('num_heads', 8),
        bottleneck_type=config['model'].get('bottleneck_type', 'attention')
    ).to(device)
    
    model_path = os.path.join("checkpoints", stem, f"best_model_{stem}.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        logger.info(f"✅ Loaded [{stem}] checkpoint")
    else:
        logger.warning(f"⚠️ Warning: [{stem}] checkpoint not found at {model_path}. Using random weights.")
    
    model.eval()
    return model

def separate(model, mix, engine, config, device):
    """
    Apply model to mixture using chunked processing.
    mix: (2, Samples)
    """
    sample_rate = config['audio']['sample_rate']
    chunk_samples = int(6.0 * sample_rate)
    overlap_samples = int(1.0 * sample_rate)
    hop_samples = chunk_samples - overlap_samples
    
    total_samples = mix.shape[1]
    out_audio = torch.zeros((2, total_samples), device='cpu')
    weight_mask = torch.zeros((1, total_samples), device='cpu')
    
    window = torch.hann_window(overlap_samples * 2)
    chunk_weight = torch.ones(chunk_samples)
    chunk_weight[:overlap_samples] = window[:overlap_samples]
    chunk_weight[-overlap_samples:] = window[overlap_samples:]

    for start in range(0, total_samples, hop_samples):
        end = min(start + chunk_samples, total_samples)
        actual_len = end - start
        if actual_len < engine.hop_length: break
        
        chunk = mix[:, start:end]
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
    return out_audio

def load_audio(path, sample_rate):
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    return audio

def evaluate(config, args):
    """
    Evaluate model using custom data loader and calculate_sdr.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = config['audio']['sample_rate']
    
    # Custom Track Discovery
    test_dir = args.musdb_path
    # If the path contains a 'test' subdirectory, use it
    if os.path.exists(os.path.join(args.musdb_path, "test")):
        test_dir = os.path.join(args.musdb_path, "test")
    
    track_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    if len(track_names) == 0:
        logger.error(f"No track directories found at {test_dir}.")
        return None, []

    logger.info(f"Found {len(track_names)} tracks in {test_dir}")

    engine = AudioEngine(
        sample_rate=sample_rate,
        n_fft=config['audio'].get('n_fft', 6144),
        hop_length=config['audio'].get('hop_length', 1024),
        win_length=config['audio'].get('win_length', 6144)
    )

    # Filter target stems
    all_stems = ["vocals", "bass", "drums", "other"]
    target_stems = all_stems if "all" in args.stems else args.stems
    
    models = {}
    for stem in target_stems:
        models[stem] = load_model(stem, config, device)

    tracks_scores = {}
    
    for track_name in tqdm(track_names, desc="Evaluating Tracks"):
        track_path = os.path.join(test_dir, track_name)
        
        # 1. Load all stems and create mixture
        stem_audio_dict = {}
        valid_track = True
        max_len = 0
        
        for s in all_stems:
            stem_file = os.path.join(track_path, f"{s}.{args.extension}")
            if not os.path.exists(stem_file):
                logger.warning(f"Missing {s} in {track_name}, skipping track.")
                valid_track = False
                break
            
            audio = load_audio(stem_file, sample_rate)
            stem_audio_dict[s] = audio
            max_len = max(max_len, audio.shape[1])
            
        if not valid_track: continue
        
        # Ensure all stems have the same length and create mixture
        mix = torch.zeros((2, max_len))
        for s in all_stems:
            audio = stem_audio_dict[s]
            if audio.shape[1] < max_len:
                audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[1]))
                stem_audio_dict[s] = audio
            mix += audio
            
        # 2. Normalization (Z-score)
        mono = mix.mean(dim=0)
        mean = mono.mean()
        std = mono.std()
        mix_norm = (mix - mean) / (std + 1e-10)
        
        track_scores = {}
        for stem in target_stems:
            # Separate
            estimates = separate(models[stem], mix_norm.to(device), engine, config, device)
            
            # Inverse normalization
            estimates = estimates * std + mean
            
            # Prepare reference
            ref = stem_audio_dict[stem]
            
            # Match lengths
            min_len = min(ref.shape[1], estimates.shape[1])
            ref = ref[:, :min_len]
            estimates = estimates[:, :min_len]
            
            # Compute SDR
            sdr_val = calculate_sdr(ref.unsqueeze(0), estimates.unsqueeze(0))
            track_scores[stem] = sdr_val.item()
            
        tracks_scores[track_name] = track_scores

    # Aggregate results
    result = {}
    actual_evaluated_tracks = list(tracks_scores.keys())
    if not actual_evaluated_tracks:
        return None, []

    for stem in target_stems:
        stem_scores = [tracks_scores[tn][stem] for tn in actual_evaluated_tracks]
        result[f"sdr_{stem}"] = np.mean(stem_scores)
        result[f"sdr_med_{stem}"] = np.median(stem_scores)
    
    all_source_means = [result[f"sdr_{stem}"] for stem in target_stems]
    all_source_medians = [result[f"sdr_med_{stem}"] for stem in target_stems]
    
    result["sdr_avg_mean"] = np.mean(all_source_means) if all_source_means else 0
    result["sdr_avg_median"] = np.mean(all_source_medians) if all_source_medians else 0

    return result, target_stems

def main():
    parser = argparse.ArgumentParser(description="Evaluate Moises-Light model using MUSDB18")
    parser.add_argument('--musdb_path', type=str, required=True, help='Path to MUSDB18 dataset')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--stems', type=str, nargs='+', default=['all'], help='Stems to evaluate (e.g., vocals bass)')
    parser.add_argument('--extension', type=str, default='flac', help='Audio file extension (default: flac)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run evaluation
    results, target_stems = evaluate(config, args)
    
    if results:
        print("\n" + "="*50)
        print(f"{'Source':<10} | {'Mean SDR':<12} | {'Median SDR':<12}")
        print("-" * 50)
        for source in target_stems:
            mean_val = results[f"sdr_{source}"]
            med_val = results[f"sdr_med_{source}"]
            print(f"{source:<10} | {mean_val:>12.4f} | {med_val:>12.4f}")
        print("-" * 50)
        print(f"{'AVERAGE':<10} | {results['sdr_avg_mean']:>12.4f} | {results['sdr_avg_median']:>12.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
