import os
import yaml
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import argparse
import sys
from tqdm import tqdm
import museval

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.roformer.model import LightRoformer
from src.utils.audio import AudioEngine

def evaluate_stem(root_dir, checkpoint_dir, config_path, target_stem, checkpoint_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = config['audio']['sample_rate']
    
    engine = AudioEngine(
        n_fft=config['audio'].get('n_fft', 6144), 
        hop_length=config['audio'].get('hop_length', 1024), 
        win_length=config['audio'].get('win_length', 6144), 
        sample_rate=sample_rate
    )
    
    # In original evaluate.py, it expects 4 stems for museval. 
    # To keep logic similar, we'll still use the 4 stems structure but only process the target.
    stems = ['vocals', 'drums', 'bass', 'other']
    if target_stem not in stems:
        print(f"Error: target_stem must be one of {stems}")
        return

    print(f"Loading model for [{target_stem}]...")
    model = LightRoformer(
        in_channels=2, out_channels=2,
        n_band=config['model'].get('num_bands', 4),
        G=config['model'].get('G', 8),
        n_layers=config['model'].get('n_rope', 5),
        n_heads=config['model'].get('num_heads', 8),
        bottleneck_type=config['model'].get('bottleneck_type', 'attention')
    ).to(device)
    
    # Resolve checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(checkpoint_dir, target_stem, f"best_model_{target_stem}.pth")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"  [OK] Loaded from {checkpoint_path}")
    else:
        print(f"  [!] Missing checkpoint at {checkpoint_path}, using random weights")
    
    model.eval()

    test_dir = os.path.join(root_dir, "test")
    if not os.path.exists(test_dir):
        # Fallback for datasets where test/ is not a subdir
        test_dir = root_dir
        
    track_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    all_scores_chunk = []
    all_scores_track = []
    duration = config['audio'].get('duration', 4.0)
    segment_length = int(duration * sample_rate)

    print(f"Evaluating {len(track_names)} tracks for stem: {target_stem}...")
    for track_name in tqdm(track_names):
        track_path = os.path.join(test_dir, track_name)
        
        # Load Mixture and Target Stem
        # To calculate SDR via museval, we ideally need the reference stem.
        target_path = os.path.join(track_path, f"{target_stem}.flac")
        if not os.path.exists(target_path):
            continue
            
        target_audio, sr = torchaudio.load(target_path)
        if sr != sample_rate:
            target_audio = torchaudio.transforms.Resample(sr, sample_rate)(target_audio)
        if target_audio.shape[0] == 1: target_audio = target_audio.repeat(2, 1)
        
        max_len = target_audio.shape[1]
        
        # In original script, it recreates mixture from all stems. 
        # Here we look for mix.flac or recreate from existing stems if available.
        mix_path = os.path.join(track_path, "mixture.flac")
        if os.path.exists(mix_path):
            mix_audio, sr = torchaudio.load(mix_path)
            if sr != sample_rate:
                mix_audio = torchaudio.transforms.Resample(sr, sample_rate)(mix_audio)
            if mix_audio.shape[0] == 1: mix_audio = mix_audio.repeat(2, 1)
            # Match length
            if mix_audio.shape[1] > max_len:
                mix_audio = mix_audio[:, :max_len]
            elif mix_audio.shape[1] < max_len:
                mix_audio = F.pad(mix_audio, (0, max_len - mix_audio.shape[1]))
        else:
            # If mixture doesn't exist, we can't evaluate accurately unless we have all stems.
            # But for single stem test, we usually assume mixture is available or target is the source of truth for length.
            # Original evaluate.py logic: sum all stems. 
            # To stay close to original, we try to find all 4 stems to make mixture.
            mix_audio = torch.zeros(2, max_len)
            for s in stems:
                s_path = os.path.join(track_path, f"{s}.flac")
                if os.path.exists(s_path):
                    s_audio, sr = torchaudio.load(s_path)
                    if sr != sample_rate:
                        s_audio = torchaudio.transforms.Resample(sr, sample_rate)(s_audio)
                    if s_audio.shape[0] == 1: s_audio = s_audio.repeat(2, 1)
                    if s_audio.shape[1] > max_len: s_audio = s_audio[:, :max_len]
                    mix_audio += F.pad(s_audio, (0, max(0, max_len - s_audio.shape[1])))[:, :max_len]

        estimate = torch.zeros_like(mix_audio)
        with torch.no_grad():
            for start in range(0, max_len, segment_length):
                end = min(start + segment_length, max_len)
                chunk = mix_audio[:, start:end]
                actual_len = chunk.shape[1]
                if actual_len < segment_length:
                    chunk = F.pad(chunk, (0, segment_length - actual_len))
                
                chunk_spec = engine.stft(chunk.unsqueeze(0).to(device))
                est_spec = model(chunk_spec)
                est_audio = engine.istft(est_spec, length=segment_length).squeeze(0).cpu()
                estimate[:, start:end] = est_audio[:, :actual_len]
        
        # museval requires (nsrc, nsampl, nchan)
        # We only have 1 source to evaluate, but museval expected 4 in original.
        # To maintain metric compatibility, we'll create a 1-source eval or pad with zeros.
        # Actually, let's just evaluate the single source.
        ref = target_audio.numpy().T[np.newaxis, ...] # (1, Samples, 2)
        est = estimate.numpy().T[np.newaxis, ...]      # (1, Samples, 2)
        
        try:
            # cSDR
            res_c = museval.evaluate(ref, est, win=int(sample_rate), hop=int(sample_rate))
            all_scores_chunk.append(res_c)
            
            # uSDR
            track_length = ref.shape[1]
            res_u = museval.evaluate(ref, est, win=track_length, hop=track_length)
            all_scores_track.append(res_u)
        except Exception as e:
            print(f"Error evaluating {track_name}: {e}")

    # Results aggregation
    print("\n" + "="*60)
    print(f"STEM: {target_stem.upper()}")
    print("-" * 60)
    
    # Aggregate cSDR
    track_medians_c = []
    for score in all_scores_chunk:
        try:
            sdr_values = score[0][0] if isinstance(score, tuple) else score.targets[0].metrics['SDR'].values
            clean_sdrs = sdr_values[~np.isnan(sdr_values)]
            if len(clean_sdrs) > 0: track_medians_c.append(np.median(clean_sdrs))
        except: continue
        
    # Aggregate uSDR
    track_sdrs_u = []
    for score in all_scores_track:
        try:
            sdr_values = score[0][0] if isinstance(score, tuple) else score.targets[0].metrics['SDR'].values
            clean_sdrs = sdr_values[~np.isnan(sdr_values)]
            if len(clean_sdrs) > 0: track_sdrs_u.append(np.mean(clean_sdrs))
        except: continue
    
    csdr_val = np.median(track_medians_c) if track_medians_c else float('nan')
    usdr_val = np.mean(track_sdrs_u) if track_sdrs_u else float('nan')
    
    print(f"cSDR (Median): {csdr_val:10.2f} dB")
    print(f"uSDR (Mean):   {usdr_val:10.2f} dB")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", type=str, required=True, choices=['vocals', 'drums', 'bass', 'other'], help="Stem to evaluate")
    parser.add_argument("--root_dir", type=str, default="data/musdb18hq_flac", help="Path to MUSDB18-HQ root")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints base directory")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Direct path to a specific .pth file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args = parser.parse_args()
    
    evaluate_stem(args.root_dir, args.checkpoint_dir, args.config, args.stem, args.checkpoint_path)
