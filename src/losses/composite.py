import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """Reference-selectable source-separation objective.

    ``scnet_spec_rmse`` is the default paper preset.  It applies a single
    complex STFT RMSE to the waveform reconstructed from the model spectrum,
    matching SCNet's optimization path. ``dttnet_waveform_l1`` and the old
    ``legacy_composite_l1`` objective are retained for controlled ablations.
    """

    def __init__(
        self,
        sample_rate=44100,
        fft_sizes=(4096, 2048, 1024, 512, 256),
        hop_length=147,
        multi_resolution_weight=1.0,
        mode="scnet_spec_rmse",
        n_fft=6144,
        win_length=None,
        normalized=True,
    ):
        super().__init__()
        valid_modes = {
            "scnet_spec_rmse",
            "dttnet_waveform_l1",
            "legacy_composite_l1",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unknown loss mode {mode!r}; expected one of {sorted(valid_modes)}")
        self.sample_rate = sample_rate
        self.fft_sizes = tuple(fft_sizes)
        self.hop_length = hop_length
        self.multi_resolution_weight = multi_resolution_weight
        self.mode = mode
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.normalized = normalized

    @staticmethod
    def _complex_mae(estimate, target):
        return F.l1_loss(torch.view_as_real(estimate), torch.view_as_real(target))

    @staticmethod
    def _complex_rmse(estimate, target):
        difference = torch.view_as_real(estimate - target)
        # Preserve batch/channel axes, as in SCNet, then average their RMSE.
        reduce_dims = tuple(range(2, difference.ndim))
        return difference.square().mean(dim=reduce_dims).sqrt().mean()

    def _stft(self, waveform, n_fft, hop_length, win_length, normalized=False):
        flattened = waveform.float().reshape(-1, waveform.shape[-1])
        window = torch.hann_window(
            win_length, device=waveform.device, dtype=torch.float32
        )
        spectrum = torch.stft(
            flattened,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=normalized,
            return_complex=True,
        )
        return spectrum.reshape(*waveform.shape[:-1], *spectrum.shape[-2:])

    def forward(self, estimates, targets, audio_engine, return_breakdown=False):
        device_type = next(iter(estimates.values())).device.type
        amp_disabled = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )
        with amp_disabled:
            waveform_loss = torch.zeros((), device=next(iter(estimates.values())).device)
            mr_loss = torch.zeros_like(waveform_loss)
            spectrum_rmse = torch.zeros_like(waveform_loss)
            zero_waveform_loss = torch.zeros_like(waveform_loss)
            zero_mr_loss = torch.zeros_like(waveform_loss)
            zero_spectrum_rmse = torch.zeros_like(waveform_loss)
            for stem, estimate_spec in estimates.items():
                target_spec = targets[stem]
                target_wave = audio_engine.istft(target_spec.to(torch.complex64))
                estimate_wave = audio_engine.istft(
                    estimate_spec.to(torch.complex64), length=target_wave.shape[-1]
                )
                waveform_loss = waveform_loss + F.l1_loss(estimate_wave, target_wave)
                zero_waveform_loss = zero_waveform_loss + target_wave.abs().mean()

                if self.mode == "scnet_spec_rmse":
                    estimate_stft = self._stft(
                        estimate_wave,
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        self.normalized,
                    )
                    target_stft = self._stft(
                        target_wave,
                        self.n_fft,
                        self.hop_length,
                        self.win_length,
                        self.normalized,
                    )
                    spectrum_rmse = spectrum_rmse + self._complex_rmse(
                        estimate_stft, target_stft
                    )
                    zero_spectrum_rmse = zero_spectrum_rmse + self._complex_rmse(
                        torch.zeros_like(target_stft), target_stft
                    )
                elif self.mode == "legacy_composite_l1":
                    for fft_size in self.fft_sizes:
                        estimate_stft = self._stft(
                            estimate_wave, fft_size, self.hop_length, fft_size
                        )
                        target_stft = self._stft(
                            target_wave, fft_size, self.hop_length, fft_size
                        )
                        mr_loss = mr_loss + self._complex_mae(
                            estimate_stft, target_stft
                        )
                        zero_mr_loss = zero_mr_loss + torch.view_as_real(
                            target_stft
                        ).abs().mean()

            count = max(1, len(estimates))
            waveform_loss = waveform_loss / count
            mr_loss = mr_loss / count
            spectrum_rmse = spectrum_rmse / count
            zero_waveform_loss = zero_waveform_loss / count
            zero_mr_loss = zero_mr_loss / count
            zero_spectrum_rmse = zero_spectrum_rmse / count
            if self.mode == "scnet_spec_rmse":
                total = spectrum_rmse
                zero_total = zero_spectrum_rmse
            elif self.mode == "dttnet_waveform_l1":
                total = waveform_loss
                zero_total = zero_waveform_loss
            else:
                total = waveform_loss + self.multi_resolution_weight * mr_loss
                zero_total = zero_waveform_loss + self.multi_resolution_weight * zero_mr_loss
        if return_breakdown:
            return total, {
                "mode": self.mode,
                "waveform_l1": waveform_loss,
                "multi_stft_mae": mr_loss,
                "spectrogram_rmse": spectrum_rmse,
                "zero_waveform_l1": zero_waveform_loss,
                "zero_multi_stft_mae": zero_mr_loss,
                "zero_spectrogram_rmse": zero_spectrum_rmse,
                "zero_total": zero_total,
            }
        return total


def spec_rmse_loss(spec_estimate, spec_sources):
    """Complex spectrum RMSE helper retained for external callers."""
    frames = min(spec_estimate.shape[-1], spec_sources.shape[-1])
    difference = torch.view_as_real(spec_estimate[..., :frames] - spec_sources[..., :frames])
    return difference.square().mean(dim=tuple(range(2, difference.ndim))).sqrt().mean()
