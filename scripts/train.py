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
    full_dataset_metadata = MUSDBDataset(
        root_dir=args.data_dir,
        sample_rate=config['audio']['sample_rate']
    )
    
    total_count = len(full_dataset_metadata)
    if total_count == 0:
        print(f"❌ Error: No tracks found at '{args.data_dir}'.")
        return

    indices = list(range(total_count))
    random.seed(42)
    random.shuffle(indices)
    
    val_size = int(total_count * 0.2)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset_full = MUSDBDataset(
        root_dir=args.data_dir, 
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio'].get('duration', 4.0),
        is_train=True
    )
    
    val_dataset_full = MUSDBDataset(
        root_dir=args.data_dir,
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio'].get('duration', 4.0),
        is_train=False
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"✅ Dataset Splitting: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Model initialization
    model = LightRoformer(
        in_channels=2,
        out_channels=2,
        n_band=config['model'].get('num_bands', 4),
        G=config['model'].get('G', 8),
        n_layers=config['model'].get('n_rope', 5),
        n_heads=config['model'].get('num_heads', 8)
    )
    
    # Run trainer
    trainer = StemTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        target_stem=args.stem
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: Total={total_params:,} | Trainable={trainable_params:,}")
    
    print(f"🚀 Starting training for [{args.stem}] stem...")
    trainer.fit()

if __name__ == "__main__":
    main()
