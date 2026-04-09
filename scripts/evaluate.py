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

def evaluate_project(root_dir, checkpoint_dir, config_path):
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
    
    stems = ['vocals', 'drums', 'bass', 'other']
    models = {}
    
    print("Loading models...")
    for stem in stems:
        model = LightRoformer(
            in_channels=2, out_channels=2,
            n_band=config['model'].get('num_bands', 4),
            G=config['model'].get('G', 8),
            n_layers=config['model'].get('n_rope', 5),
            n_heads=config['model'].get('num_heads', 8)
        ).to(device)
        
        model_path = os.path.join(checkpoint_dir, stem, f"best_model_{stem}.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print(f"  [OK] Loaded {stem}")
        else:
            print(f"  [!] Missing {stem}, using random weights")
        model.eval()
        models[stem] = model

    test_dir = os.path.join(root_dir, "test")
    track_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    all_scores_chunk = []
    all_scores_track = []
    duration = config['audio'].get('duration', 4.0)
    segment_length = int(duration * sample_rate)

    print(f"Evaluating {len(track_names)} tracks...")
    for track_name in tqdm(track_names):
        track_path = os.path.join(test_dir, track_name)
        stem_audio_dict = {}
        max_len = 0
        
        for stem in stems:
            stem_path = os.path.join(track_path, f"{stem}.flac")
            if os.path.exists(stem_path):
                audio, sr = torchaudio.load(stem_path)
                if sr != sample_rate:
                    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
                if audio.shape[0] == 1: audio = audio.repeat(2, 1)
                stem_audio_dict[stem] = audio
                max_len = max(max_len, audio.shape[1])
        
        mix_audio = torch.zeros(2, max_len)
        targets_list = []
        for stem in stems:
            if stem in stem_audio_dict:
                audio = stem_audio_dict[stem]
                if audio.shape[1] < max_len:
                    audio = F.pad(audio, (0, max_len - audio.shape[1]))
                mix_audio += audio
                targets_list.append(audio.numpy().T)
            else:
                targets_list.append(np.zeros((max_len, 2)))
        
        track_estimates = {s: torch.zeros_like(mix_audio) for s in stems}
        with torch.no_grad():
            for start in range(0, max_len, segment_length):
                end = min(start + segment_length, max_len)
                chunk = mix_audio[:, start:end]
                actual_len = chunk.shape[1]
                if actual_len < segment_length:
                    chunk = F.pad(chunk, (0, segment_length - actual_len))
                
                chunk_spec = engine.stft(chunk.unsqueeze(0).to(device))
                for stem in stems:
                    est_spec = models[stem](chunk_spec)
                    est_audio = engine.istft(est_spec, length=segment_length).squeeze(0).cpu()
                    track_estimates[stem][:, start:end] = est_audio[:, :actual_len]
        
        ref = np.stack(targets_list, axis=0).astype(np.float64)
        est = np.stack([track_estimates[s].numpy().T for s in stems], axis=0).astype(np.float64)
        
        try:
            # cSDR: Calculate per-second window
            # Ensure win and hop are integers as museval/mir_eval might be sensitive to types
            res_c = museval.evaluate(ref, est, win=int(sample_rate), hop=int(sample_rate))
            all_scores_chunk.append(res_c)
            
            # uSDR: Calculate per-track (Use full length to avoid NoneType comparison error)
            track_length = ref.shape[1]
            res_u = museval.evaluate(ref, est, win=track_length, hop=track_length)
            all_scores_track.append(res_u)
        except Exception as e:
            print(f"Error evaluating {track_name}: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace for better debugging

    # Results aggregation
    print("\n" + "="*60)
    print(f"{'STEM':<10} | {'cSDR (Median)':<15} | {'uSDR (Mean)':<15}")
    print("-" * 60)
    
    for i, stem in enumerate(stems):
        # Aggregate cSDR (Median of Medians)
        track_medians_c = []
        for score in all_scores_chunk:
            try:
                sdr_values = score[0][i] if isinstance(score, tuple) else score.targets[i].metrics['SDR'].values
                clean_sdrs = sdr_values[~np.isnan(sdr_values)]
                if len(clean_sdrs) > 0: track_medians_c.append(np.median(clean_sdrs))
            except: continue
            
        # Aggregate uSDR (Mean of Track SDRs)
        track_sdrs_u = []
        for score in all_scores_track:
            try:
                sdr_values = score[0][i] if isinstance(score, tuple) else score.targets[i].metrics['SDR'].values
                clean_sdrs = sdr_values[~np.isnan(sdr_values)]
                if len(clean_sdrs) > 0: track_sdrs_u.append(np.mean(clean_sdrs))
            except: continue
        
        csdr_val = np.median(track_medians_c) if track_medians_c else float('nan')
        usdr_val = np.mean(track_sdrs_u) if track_sdrs_u else float('nan')
        
        print(f"{stem.upper():<10} | {csdr_val:10.2f} dB | {usdr_val:10.2f} dB")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/musdb18hq_flac", help="Path to MUSDB18-HQ root")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args = parser.parse_args()
    evaluate_project(args.root_dir, args.checkpoint_dir, args.config)
