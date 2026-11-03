from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SDRResult:
    usdr: float
    csdr: float
    chunk_scores: np.ndarray


def direct_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8):
    """Unfiltered utterance SDR used by the DTTNet/Moises-Light protocol."""
    if reference.shape != estimate.shape:
        raise ValueError(f"SDR shape mismatch: {reference.shape} vs {estimate.shape}")
    reduce_dims = tuple(range(1, reference.ndim))
    signal = reference.float().square().sum(dim=reduce_dims)
    distortion = (reference.float() - estimate.float()).square().sum(dim=reduce_dims)
    return 10.0 * torch.log10((signal + eps) / (distortion + eps))


def chunk_sdr(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    chunk_size: int,
    eps: float = 1e-8,
    silence_threshold: float = 1e-8,
):
    """Direct SDR on non-overlapping chunks; silent references are reported as NaN."""
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
        estimate = estimate.unsqueeze(0)
    if reference.shape != estimate.shape:
        raise ValueError(f"SDR shape mismatch: {reference.shape} vs {estimate.shape}")
    scores = []
    total = reference.shape[-1]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if end - start < chunk_size:
            break
        ref_chunk = reference[..., start:end].float()
        est_chunk = estimate[..., start:end].float()
        signal = ref_chunk.square().sum(dim=tuple(range(1, ref_chunk.ndim)))
        distortion = (ref_chunk - est_chunk).square().sum(
            dim=tuple(range(1, ref_chunk.ndim))
        )
        score = 10.0 * torch.log10((signal + eps) / (distortion + eps))
        score = torch.where(
            signal > silence_threshold,
            score,
            torch.full_like(score, torch.nan),
        )
        scores.append(score)
    if not scores:
        return torch.empty(0, device=reference.device)
    return torch.cat(scores)


def compute_sdr_metrics(reference, estimate, sample_rate=44100, chunk_seconds=1.0):
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
        estimate = estimate.unsqueeze(0)
    usdr = direct_sdr(reference, estimate).mean().item()
    chunks = chunk_sdr(
        reference,
        estimate,
        chunk_size=max(1, int(sample_rate * chunk_seconds)),
    )
    chunks_np = chunks.detach().cpu().numpy()
    csdr = float(np.nanmedian(chunks_np)) if chunks_np.size else float("nan")
    return SDRResult(usdr=usdr, csdr=csdr, chunk_scores=chunks_np)


def aggregate_track_metrics(results):
    """Paper reporting: mean track uSDR and median of per-track cSDR medians."""
    if not results:
        raise ValueError("No valid tracks were available for SDR aggregation")
    usdr = float(np.mean([result.usdr for result in results]))
    csdr = float(np.nanmedian([result.csdr for result in results]))
    return {"usdr_mean": usdr, "csdr_median": csdr}
