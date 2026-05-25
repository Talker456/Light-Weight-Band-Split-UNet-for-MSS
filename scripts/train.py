import yaml
import torch
import argparse
import os
import random
import sys

# Add project root to sys.path to recognize src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Subset
from src.models.roformer.model import LightRoformer
from src.data.dataset import MUSDBDataset
from src.training.trainer import StemTrainer

def main():
    parser = argparse.ArgumentParser(description="Moises-Light Roformer Unified Training Script")
    parser.add_argument("--stem", type=str, default="vocals", 
                        choices=["vocals", "bass", "drums", "other"],
                        help="Target stem to train (vocals, bass, drums, other)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--data_dir", type=str, default="data/musdb18hq_flac/train",
                        help="Path to the training data directory")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target Stem: [{args.stem}]")
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directory '{args.data_dir}' not found.")
        return

    # Dataset initialization and splitting (80:20)
    temp_dataset = MUSDBDataset(
        root_dir=args.data_dir,
        sample_rate=config['audio']['sample_rate']
    )
    
    all_tracks = temp_dataset.tracks
    if not all_tracks:
        print(f"❌ Error: No tracks found at '{args.data_dir}'.")
        return

    random.seed(42)
    random.shuffle(all_tracks)
    
    val_size = int(len(all_tracks) * 0.1)
    val_tracks = all_tracks[:val_size]
    train_tracks = all_tracks[val_size:]

    train_dataset = MUSDBDataset(
        root_dir=args.data_dir, 
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio'].get('duration', 4.0),
        is_train=True,
        samples_per_track=config['training'].get('samples_per_track', 1),
        tracks=train_tracks
    )
    
    val_dataset = MUSDBDataset(
        root_dir=args.data_dir,
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio'].get('duration', 4.0),
        is_train=False,
        tracks=val_tracks
    )

    print(f"✅ Dataset Splitting: Train={len(train_dataset)} steps, Val={len(val_dataset)} tracks")

    # Model initialization
    model = LightRoformer(
        in_channels=2,
        out_channels=2,
        n_band=config['model'].get('num_bands', 4),
        G=config['model'].get('G', 8),
        n_layers=config['model'].get('n_rope', 6),
        n_heads=config['model'].get('num_heads', 8),
        bottleneck_type=config['model'].get('bottleneck_type', 'rnn')
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: Total={total_params:,} | Trainable={trainable_params:,}")
    
    # Run trainer
    trainer = StemTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        target_stem=args.stem
    )
    
    print(f"🚀 Starting training for [{args.stem}] stem...")
    trainer.fit()

if __name__ == "__main__":
    main()
