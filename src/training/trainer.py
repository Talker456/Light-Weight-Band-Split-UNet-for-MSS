import json
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.composite import CompositeLoss
from src.utils.audio import AudioEngine
from src.utils.inference import apply_model_to_spec
from src.utils.metrics import aggregate_track_metrics, compute_sdr_metrics


class StemTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, device, target_stem="vocals"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.target_stem = target_stem
        self.sample_rate = config["audio"]["sample_rate"]
        self.audio_engine = AudioEngine(
            sample_rate=self.sample_rate,
            n_fft=config["audio"].get("n_fft", 6144),
            hop_length=config["audio"].get("hop_length", 1024),
            win_length=config["audio"].get("win_length", 6144),
        )
        duration = config["audio"].get("duration", 4.0)
        dummy = torch.zeros(1, 1, int(self.sample_rate * duration))
        self.segment_frames = self.audio_engine.stft(dummy).shape[-1]

        workers = config["training"].get("num_workers", 2)
        loader_options = {
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
        }
        if workers > 0:
            loader_options["persistent_workers"] = config["training"].get(
                "persistent_workers", False
            )
            loader_options["prefetch_factor"] = config["training"].get(
                "prefetch_factor", 2
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"].get("micro_batch_size", config["training"]["batch_size"]),
            shuffle=True,
            **loader_options,
        )
        # Whole-song tensors are large; avoiding worker IPC prevents Colab shared-
        # memory exhaustion during validation.
        val_workers = config["training"].get("val_num_workers", 0)
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=val_workers
        )
        optimizer_name = config["training"].get("optimizer", "adamw").lower()
        optimizer_class = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError("training.optimizer must be 'adam' or 'adamw'")
        self.optimizer = optimizer_class(
            model.parameters(),
            lr=float(config["training"]["lr"]),
            weight_decay=config["training"].get("weight_decay", 0.0),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["training"]["epochs"]
        )
        loss_config = config.get("loss", {})
        self.criterion = CompositeLoss(
            sample_rate=self.sample_rate,
            fft_sizes=tuple(loss_config.get("fft_sizes", [4096, 2048, 1024, 512, 256])),
            hop_length=loss_config.get(
                "hop_length", config["audio"].get("hop_length", 1024)
            ),
            multi_resolution_weight=loss_config.get("multi_resolution_weight", 1.0),
            mode=loss_config.get("mode", "scnet_spec_rmse"),
            n_fft=loss_config.get("n_fft", config["audio"].get("n_fft", 6144)),
            win_length=loss_config.get(
                "win_length", config["audio"].get("win_length", 6144)
            ),
            normalized=loss_config.get("normalized", True),
        ).to(device)
        self.loss_mode = self.criterion.mode
        self.epochs = config["training"]["epochs"]
        self.accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.amp_enabled = device.type == "cuda" and config["training"].get("amp", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.best_val_csdr = float("-inf")
        self.diagnostics_interval = config["training"].get(
            "diagnostics_log_interval", 50
        )
        self.log_validation_tracks = config["training"].get(
            "diagnostics_log_validation_tracks", True
        )
        self.decoder_diagnostics_interval = config["training"].get(
            "decoder_diagnostics_interval", 0
        )

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled)

    def _validation_autocast(self):
        """Validation defaults to FP32 to protect multiplicative skip fusion."""
        if self.device.type != "cuda":
            return nullcontext()
        enabled = self.config["training"].get("validation_amp", False)
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        totals = {
            "loss": 0.0,
            "waveform_l1": 0.0,
            "multi_stft_mae": 0.0,
            "spectrogram_rmse": 0.0,
            "zero_total": 0.0,
            "spec_estimate_energy": 0.0,
            "spec_target_energy": 0.0,
        }
        gradient_norms = []
        output_gradient_norms = []
        consistency_errors = []
        clipped_updates = 0
        processed_batches = 0
        skipped_nonfinite_input = 0
        skipped_nonfinite_loss = 0
        update_count = 0
        decoder_diagnostics = []
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [{self.target_stem}]")
        for batch_idx, (mixture_spec, targets) in enumerate(progress):
            mixture_spec = mixture_spec.to(self.device, non_blocking=True)
            target_spec = targets[self.target_stem].to(self.device, non_blocking=True)
            if not torch.isfinite(mixture_spec).all():
                skipped_nonfinite_input += 1
                continue
            capture_decoder = bool(
                self.decoder_diagnostics_interval
                and (batch_idx + 1) % self.decoder_diagnostics_interval == 0
            )
            decoder = getattr(self.model, "decoder", None)
            if decoder is not None:
                decoder.capture_diagnostics = capture_decoder
            with self._autocast():
                estimate = self.model(mixture_spec)
                loss, breakdown = self.criterion(
                    {self.target_stem: estimate},
                    {self.target_stem: target_spec},
                    self.audio_engine,
                    return_breakdown=True,
                )
                scaled_loss = loss / self.accumulation_steps
            if capture_decoder and decoder is not None:
                snapshot = {
                    "batch": batch_idx + 1,
                    "stages": decoder.last_diagnostics,
                }
                decoder_diagnostics.append(snapshot)
                for stage in snapshot["stages"]:
                    output = stage.get("output", {})
                    stage_input = stage.get("input", {})
                    normalized = stage.get("normalization", {})
                    output_rms = output.get("rms", float("inf"))
                    input_rms = stage_input.get("rms", 0.0)
                    rms_drop_db = 20.0 * np.log10(
                        (output_rms + 1e-12) / (input_rms + 1e-12)
                    )
                    outlier_ratio = normalized.get("abs_max", 0.0) / max(
                        normalized.get("abs_p99", 0.0), 1e-12
                    )
                    if (
                        output_rms < 1e-4
                        or output.get("nonfinite", 0)
                        or rms_drop_db < -40.0
                        or outlier_ratio > 100.0
                    ):
                        tqdm.write(
                            f"WARNING decoder stage {stage['stage']} at batch "
                            f"{batch_idx + 1}: RMS={output_rms:.3e} "
                            f"drop={rms_drop_db:.1f}dB outlier={outlier_ratio:.1f}x "
                            f"nonfinite={output.get('nonfinite')}"
                        )
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_loss += 1
                continue
            self.scaler.scale(scaled_loss).backward()
            is_update = (batch_idx + 1) % self.accumulation_steps == 0 or (
                batch_idx + 1 == len(self.train_loader)
            )
            if is_update:
                self.scaler.unscale_(self.optimizer)
                output_gradient_squared = sum(
                    parameter.grad.detach().float().square().sum().item()
                    for parameter in self.model.final_stage.parameters()
                    if parameter.grad is not None
                )
                output_gradient_norms.append(output_gradient_squared ** 0.5)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                gradient_norm_value = float(gradient_norm.detach().cpu())
                gradient_norms.append(gradient_norm_value)
                clipped_updates += int(gradient_norm_value > 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                update_count += 1
            estimate_detached = estimate.detach()
            target_detached = target_spec.detach()
            estimate_energy = (
                estimate_detached.real.float().square()
                + estimate_detached.imag.float().square()
            ).mean().item()
            target_energy = (
                target_detached.real.float().square()
                + target_detached.imag.float().square()
            ).mean().item()
            processed_batches += 1
            totals["loss"] += loss.item()
            totals["waveform_l1"] += breakdown["waveform_l1"].item()
            totals["multi_stft_mae"] += breakdown["multi_stft_mae"].item()
            totals["spectrogram_rmse"] += breakdown["spectrogram_rmse"].item()
            totals["zero_total"] += breakdown["zero_total"].item()
            totals["spec_estimate_energy"] += estimate_energy
            totals["spec_target_energy"] += target_energy
            log_diagnostics = (batch_idx + 1) % self.diagnostics_interval == 0 or (
                is_update and batch_idx < 4
            )
            if log_diagnostics:
                with torch.no_grad():
                    estimate_wave = self.audio_engine.istft(estimate_detached)
                    projected_spec = self.audio_engine.stft(estimate_wave)
                    frames = min(projected_spec.shape[-1], estimate_detached.shape[-1])
                    projected_spec = projected_spec[..., :frames]
                    raw_spec = estimate_detached[..., :frames]
                    consistency_error = (
                        (projected_spec - raw_spec).abs().square().sum().sqrt()
                        / raw_spec.abs().square().sum().sqrt().clamp_min(1e-12)
                    ).item()
                    consistency_errors.append(consistency_error)
                silence_ratio = loss.item() / max(breakdown["zero_total"].item(), 1e-12)
                spec_gain_db = 10.0 * np.log10(
                    (estimate_energy + 1e-12) / (target_energy + 1e-12)
                )
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    zero=f"{silence_ratio:.3f}x",
                    spec_gain=f"{spec_gain_db:.1f}dB",
                    consistency=f"{consistency_error:.3f}",
                    grad=f"{gradient_norms[-1]:.3g}" if gradient_norms else "n/a",
                    updates=update_count,
                )

        count = max(1, processed_batches)
        metrics = {key: value / count for key, value in totals.items()}
        metrics["loss_vs_silence"] = metrics["loss"] / max(
            metrics["zero_total"], 1e-12
        )
        metrics["spec_gain_db"] = 10.0 * np.log10(
            (metrics["spec_estimate_energy"] + 1e-12)
            / (metrics["spec_target_energy"] + 1e-12)
        )
        metrics["gradient_norm_mean"] = (
            float(np.mean(gradient_norms)) if gradient_norms else float("nan")
        )
        metrics["gradient_norm_max"] = (
            float(np.max(gradient_norms)) if gradient_norms else float("nan")
        )
        metrics["output_gradient_norm_mean"] = (
            float(np.mean(output_gradient_norms))
            if output_gradient_norms
            else float("nan")
        )
        metrics["consistency_error_mean"] = (
            float(np.mean(consistency_errors))
            if consistency_errors
            else float("nan")
        )
        metrics["loss_mode"] = self.loss_mode
        metrics["clipped_update_fraction"] = clipped_updates / max(
            1, len(gradient_norms)
        )
        metrics["processed_batches"] = processed_batches
        metrics["skipped_nonfinite_input"] = skipped_nonfinite_input
        metrics["skipped_nonfinite_loss"] = skipped_nonfinite_loss
        metrics["optimizer_updates"] = update_count
        metrics["amp_scale"] = float(self.scaler.get_scale())
        metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["parameter_l2"] = float(
            sum(
                parameter.detach().float().square().sum().item()
                for parameter in self.model.parameters()
            )
            ** 0.5
        )
        metrics["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == "cuda"
            else 0.0
        )
        metrics["decoder_diagnostics"] = decoder_diagnostics
        return metrics

    def validate(self):
        self.model.eval()
        overlap = self.config["training"].get("val_overlap", 0.5)
        results = []
        losses = []
        silence_losses = []
        total_target_energy = 0.0
        total_estimate_energy = 0.0
        total_cross = 0.0
        total_samples = 0
        track_diagnostics = []
        with torch.no_grad():
            for track_index, (mixture_spec, targets) in enumerate(
                tqdm(self.val_loader, desc=f"Valid [{self.target_stem}]")
            ):
                # Keep whole-track tensors on CPU. Only one spectral window at a
                # time is moved to the T4; completed windows are accumulated on CPU.
                mixture_spec = mixture_spec.squeeze(0)
                target_spec = targets[self.target_stem]
                with self._validation_autocast():
                    estimate = apply_model_to_spec(
                        self.model,
                        mixture_spec,
                        self.segment_frames,
                        overlap=overlap,
                        model_device=self.device,
                        output_device=torch.device("cpu"),
                    ).unsqueeze(0)
                estimate_finite = torch.isfinite(estimate.real).all() and torch.isfinite(
                    estimate.imag
                ).all()
                if not bool(estimate_finite):
                    track_name = self.val_loader.dataset.tracks[track_index]
                    raise FloatingPointError(
                        f"Non-finite validation estimate for {track_name!r}. "
                        "Validation checkpoints and SDR metrics were not updated."
                    )
                target_wave = self.audio_engine.istft(target_spec)
                estimate_wave = self.audio_engine.istft(estimate, length=target_wave.shape[-1])
                # A full multi-resolution validation loss would materialize five
                # whole-song STFT graphs. Waveform L1 is a bounded-memory monitor;
                # checkpoint selection remains paper-faithful cSDR below.
                loss = torch.nn.functional.l1_loss(estimate_wave, target_wave)
                silence_loss = target_wave.abs().mean()
                result = compute_sdr_metrics(
                    target_wave.squeeze(0).cpu(),
                    estimate_wave.squeeze(0).cpu(),
                    sample_rate=self.sample_rate,
                )
                results.append(result)
                losses.append(loss.item())
                silence_losses.append(silence_loss.item())
                target_energy = target_wave.double().square().sum().item()
                estimate_energy = estimate_wave.double().square().sum().item()
                cross = (target_wave.double() * estimate_wave.double()).sum().item()
                total_target_energy += target_energy
                total_estimate_energy += estimate_energy
                total_cross += cross
                total_samples += target_wave.numel()
                target_rms = target_wave.double().square().mean().sqrt().item()
                estimate_rms = estimate_wave.double().square().mean().sqrt().item()
                energy_gain_db = 10.0 * np.log10(
                    (estimate_energy + 1e-12) / (target_energy + 1e-12)
                )
                correlation = cross / max(
                    (target_energy * estimate_energy) ** 0.5, 1e-12
                )
                track_name = str(track_index)
                dataset_tracks = getattr(self.val_loader.dataset, "tracks", None)
                if dataset_tracks and track_index < len(dataset_tracks):
                    track_name = dataset_tracks[track_index]
                diagnostic = {
                    "track": track_name,
                    "loss": loss.item(),
                    "loss_vs_silence": loss.item() / max(silence_loss.item(), 1e-12),
                    "target_rms": target_rms,
                    "estimate_rms": estimate_rms,
                    "energy_gain_db": energy_gain_db,
                    "correlation": correlation,
                    "usdr": result.usdr,
                    "csdr": result.csdr,
                }
                track_diagnostics.append(diagnostic)
                if self.log_validation_tracks:
                    tqdm.write(
                        f"Val [{track_name}] loss/zero={diagnostic['loss_vs_silence']:.4f} "
                        f"RMS={estimate_rms:.3e}/{target_rms:.3e} "
                        f"gain={energy_gain_db:.2f} dB corr={correlation:.4f} "
                        f"uSDR={result.usdr:.5f} cSDR={result.csdr:.5f}"
                    )
        metrics = aggregate_track_metrics(results)
        metrics.update(
            {
                "loss": float(np.mean(losses)),
                "zero_loss": float(np.mean(silence_losses)),
                "loss_vs_silence": float(np.mean(losses))
                / max(float(np.mean(silence_losses)), 1e-12),
                "target_rms": (total_target_energy / max(1, total_samples)) ** 0.5,
                "estimate_rms": (total_estimate_energy / max(1, total_samples)) ** 0.5,
                "energy_gain_db": 10.0 * np.log10(
                    (total_estimate_energy + 1e-12) / (total_target_energy + 1e-12)
                ),
                "correlation": total_cross
                / max((total_target_energy * total_estimate_energy) ** 0.5, 1e-12),
                "tracks": track_diagnostics,
            }
        )
        return metrics

    def _save_checkpoint(self, path, epoch, train_metrics, val_metrics):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_csdr": self.best_val_csdr,
            "val_loss": val_metrics["loss"],
            "val_usdr": val_metrics["usdr_mean"],
            "val_csdr": val_metrics["csdr_median"],
            "train_diagnostics": train_metrics,
            "validation_diagnostics": val_metrics,
            "model_metadata": self.model.architecture_metadata(),
            "loss_mode": self.loss_mode,
            "config": self.config,
        }
        temporary = path + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)

        backup_root = self.config["training"].get("backup_checkpoint_dir")
        if backup_root:
            relative = os.path.relpath(path, self.config["training"].get(
                "checkpoint_dir", "checkpoints/paper_v5"
            ))
            backup_path = os.path.join(backup_root, relative)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            backup_temporary = backup_path + ".tmp"
            shutil.copy2(path, backup_temporary)
            os.replace(backup_temporary, backup_path)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {key: StemTrainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [StemTrainer._json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    def _append_epoch_log(self, path, epoch, train_metrics, val_metrics):
        record = self._json_safe(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _backup_path(self, local_path):
        backup_root = self.config["training"].get("backup_checkpoint_dir")
        if not backup_root:
            return None
        checkpoint_root = self.config["training"].get(
            "checkpoint_dir", "checkpoints/paper_v5"
        )
        return os.path.join(backup_root, os.path.relpath(local_path, checkpoint_root))

    def fit(self):
        checkpoint_root = self.config["training"].get("checkpoint_dir", "checkpoints/paper_v5")
        checkpoint_dir = os.path.join(checkpoint_root, self.target_stem)
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_path = os.path.join(checkpoint_dir, f"latest_model_{self.target_stem}.pth")
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.jsonl")
        start_epoch = 1
        backup_latest_path = self._backup_path(latest_path)
        resume_path = latest_path
        if not os.path.exists(resume_path) and backup_latest_path and os.path.exists(
            backup_latest_path
        ):
            resume_path = backup_latest_path
        if os.path.exists(resume_path):
            checkpoint_data = torch.load(resume_path, map_location=self.device)
            checkpoint_loss_mode = checkpoint_data.get("loss_mode")
            if checkpoint_loss_mode != self.loss_mode:
                raise RuntimeError(
                    "Checkpoint objective is incompatible with this run: "
                    f"checkpoint={checkpoint_loss_mode!r}, config={self.loss_mode!r}. "
                    "Use a fresh checkpoint directory for a loss ablation."
                )
            metadata = checkpoint_data.get("model_metadata", {})
            self.model.validate_checkpoint_metadata(metadata)
            self.model.load_state_dict(checkpoint_data["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint_data.get("scaler_state_dict", {}))
            self.best_val_csdr = checkpoint_data.get("best_val_csdr", float("-inf"))
            start_epoch = checkpoint_data["epoch"] + 1

        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            print(
                f"Epoch {epoch}: train={train_metrics['loss']:.6f} "
                f"(objective={train_metrics['loss_mode']}, "
                f"wave={train_metrics['waveform_l1']:.6f}, "
                f"spec_rmse={train_metrics['spectrogram_rmse']:.6f}, "
                f"legacy_mr={train_metrics['multi_stft_mae']:.6f}, "
                f"loss/zero={train_metrics['loss_vs_silence']:.4f}, "
                f"spec_gain={train_metrics['spec_gain_db']:.2f} dB, "
                f"consistency={train_metrics['consistency_error_mean']:.4f}, "
                f"grad={train_metrics['gradient_norm_mean']:.4g}/"
                f"{train_metrics['gradient_norm_max']:.4g}, "
                f"out_grad={train_metrics['output_gradient_norm_mean']:.4g}, "
                f"clip={train_metrics['clipped_update_fraction']:.1%}, "
                f"amp={train_metrics['amp_scale']:.0f}, "
                f"lr={train_metrics['learning_rate']:.3e}, "
                f"skip={train_metrics['skipped_nonfinite_input']}/"
                f"{train_metrics['skipped_nonfinite_loss']}, "
                f"peak={train_metrics['peak_allocated_gib']:.2f} GiB)"
            )
            print(
                f"Epoch {epoch}: val_wave_l1={val_metrics['loss']:.6f} "
                f"(loss/zero={val_metrics['loss_vs_silence']:.4f}, "
                f"RMS={val_metrics['estimate_rms']:.3e}/"
                f"{val_metrics['target_rms']:.3e}, "
                f"gain={val_metrics['energy_gain_db']:.2f} dB, "
                f"corr={val_metrics['correlation']:.5f}), "
                f"uSDR={val_metrics['usdr_mean']:.5f} dB, "
                f"cSDR={val_metrics['csdr_median']:.5f} dB"
            )
            self._append_epoch_log(metrics_path, epoch, train_metrics, val_metrics)
            checkpoint_metric = val_metrics["csdr_median"]
            if np.isfinite(checkpoint_metric) and checkpoint_metric > self.best_val_csdr:
                self.best_val_csdr = val_metrics["csdr_median"]
                best_path = os.path.join(checkpoint_dir, f"best_model_{self.target_stem}.pth")
                self._save_checkpoint(
                    best_path, epoch, train_metrics, val_metrics
                )
            self._save_checkpoint(latest_path, epoch, train_metrics, val_metrics)
