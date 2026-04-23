import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.audio import AudioEngine
from src.losses.composite import CompositeLoss

from torch.cuda.amp import autocast, GradScaler

class StemTrainer:
    """
    Integrated trainer supporting the 'One Model per Stem' training strategy of Moises-Light Roformer.
    Integrated with Mixed Precision (AMP) for T4 acceleration.
    """
    def __init__(self, model, train_dataset, val_dataset, config, device, target_stem='vocals'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.target_stem = target_stem
        
        self.sample_rate = config['audio']['sample_rate']
        self.audio_engine = AudioEngine(sample_rate=self.sample_rate)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            num_workers=config['training'].get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False,
            num_workers=config['training'].get('num_workers', 2)
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training']['lr']),
            weight_decay=config['training'].get('weight_decay', 1e-2)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs']
        )
        
        self.criterion = CompositeLoss(sample_rate=self.sample_rate).to(self.device)
        self.epochs = config['training']['epochs']
        self.best_val_loss = float('inf')
        
        # AMP Scaler (using recommended torch.amp API)
        self.scaler = torch.amp.GradScaler(device_type='cuda')

    def _get_target_dict(self, targets):
        if self.target_stem in targets:
            return {self.target_stem: targets[self.target_stem].to(self.device)}
        else:
            raise KeyError(f"Target stem '{self.target_stem}' not found in dataset.")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train:{self.target_stem}]")
        
        for batch_idx, (mixture_spec, targets) in enumerate(pbar):
            mixture_spec = mixture_spec.to(self.device)
            target_dict = self._get_target_dict(targets)
            
            if not torch.isfinite(mixture_spec).all():
                continue

            self.optimizer.zero_grad()
            
            # Forward pass with AMP autocast (using recommended torch.amp API)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                out = self.model(mixture_spec)
                estimates = {self.target_stem: out}
                loss = self.criterion(estimates, target_dict, self.audio_engine)

            if torch.isnan(loss) or loss.item() > 20.0:
                self.optimizer.zero_grad()
                del out, loss, estimates, target_dict, mixture_spec
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Scaled Backward pass
            self.scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Scaler step and update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            del out, loss, estimates, target_dict, mixture_spec
            if batch_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True):
                for mixture_spec, targets in self.val_loader:
                    mixture_spec = mixture_spec.to(self.device)
                    target_dict = self._get_target_dict(targets)
                    out = self.model(mixture_spec)
                    estimates = {self.target_stem: out}
                    loss = self.criterion(estimates, target_dict, self.audio_engine)
                    total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        ckpt_dir = os.path.join("checkpoints", self.target_stem)
        os.makedirs(ckpt_dir, exist_ok=True)
        latest_path = os.path.join(ckpt_dir, f"latest_model_{self.target_stem}.pth")
        start_epoch = 1

        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(ckpt_dir, f"best_model_{self.target_stem}.pth")
                torch.save(self.model.state_dict(), save_path)
                
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss
            }
            torch.save(checkpoint, latest_path)
