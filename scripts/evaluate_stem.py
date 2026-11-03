import argparse
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.factory import build_model
from src.utils.audio import AudioEngine
from src.utils.inference import apply_model_to_spec
from src.utils.metrics import aggregate_track_metrics, compute_sdr_metrics


STEMS = ("vocals", "drums", "bass", "other")


def find_audio(track_path, stem):
    for extension in ("flac", "wav"):
        candidate = os.path.join(track_path, f"{stem}.{extension}")
        if os.path.exists(candidate):
            return candidate
    return None


def load_stereo(path, sample_rate):
    audio, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        audio = torchaudio.transforms.Resample(source_rate, sample_rate)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    return audio[:2]


def load_track(track_path, target_stem, sample_rate):
    target_path = find_audio(track_path, target_stem)
    if target_path is None:
        return None, None
    target = load_stereo(target_path, sample_rate)
    mixture_path = find_audio(track_path, "mixture")
    if mixture_path:
        mixture = load_stereo(mixture_path, sample_rate)
    else:
        sources = []
        for stem in STEMS:
            path = find_audio(track_path, stem)
            if path is None:
                return None, None
            sources.append(load_stereo(path, sample_rate))
        length = max(source.shape[-1] for source in sources)
        mixture = sum(F.pad(source, (0, length - source.shape[-1])) for source in sources)
    length = min(mixture.shape[-1], target.shape[-1])
    return mixture[..., :length], target[..., :length]


def evaluate_stem(root_dir, checkpoint_dir, config_path, target_stem, checkpoint_path=None):
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = config["audio"]["sample_rate"]
    engine = AudioEngine(
        n_fft=config["audio"].get("n_fft", 6144),
        hop_length=config["audio"].get("hop_length", 1024),
        win_length=config["audio"].get("win_length", 6144),
        sample_rate=sample_rate,
    )
    model = build_model(config).to(device)
    checkpoint_root = checkpoint_dir or config["training"].get(
        "checkpoint_dir", "checkpoints/paper_v5"
    )
    checkpoint_path = checkpoint_path or os.path.join(
        checkpoint_root, target_stem, f"best_model_{target_stem}.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("model_metadata", {}) if isinstance(checkpoint, dict) else {}
    model.validate_checkpoint_metadata(metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dir = os.path.join(root_dir, "test")
    if not os.path.isdir(test_dir):
        test_dir = root_dir
    tracks = sorted(
        entry for entry in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, entry))
    )
    segment_samples = int(config["audio"].get("duration", 4.0) * sample_rate)
    segment_frames = engine.stft(torch.zeros(1, 1, segment_samples)).shape[-1]
    overlap = config.get("evaluation", {}).get("overlap", 0.5)
    results = []

    with torch.no_grad():
        for track in tqdm(tracks, desc=f"Evaluate {target_stem}"):
            mixture, target = load_track(os.path.join(test_dir, track), target_stem, sample_rate)
            if mixture is None:
                continue
            mixture_spec = engine.stft(mixture.unsqueeze(0).to(device)).squeeze(0)
            evaluation_amp = device.type == "cuda" and config.get("evaluation", {}).get(
                "amp", False
            )
            precision_context = (
                torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
                if evaluation_amp
                else nullcontext()
            )
            with precision_context:
                estimate_spec = apply_model_to_spec(
                    model, mixture_spec, segment_frames, overlap=overlap
                )
            if not bool(
                torch.isfinite(estimate_spec.real).all()
                and torch.isfinite(estimate_spec.imag).all()
            ):
                raise FloatingPointError(
                    f"Non-finite estimate for {track!r}; SDR was not computed."
                )
            estimate = engine.istft(
                estimate_spec.unsqueeze(0), length=target.shape[-1]
            ).squeeze(0).cpu()
            result = compute_sdr_metrics(target, estimate, sample_rate=sample_rate)
            results.append(result)
            print(f"{track}: cSDR={result.csdr:.2f}, uSDR={result.usdr:.2f}")

    aggregate = aggregate_track_metrics(results)
    print(f"{target_stem.upper()} cSDR (median): {aggregate['csdr_median']:.2f} dB")
    print(f"{target_stem.upper()} uSDR (mean):   {aggregate['usdr_mean']:.2f} dB")
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True, choices=STEMS)
    parser.add_argument("--root_dir", default="data/musdb18hq_flac")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--config",
        default="configs/v5_proposed_s_scnet_seg9_mb1_ga4_half_epoch.yaml",
    )
    arguments = parser.parse_args()
    evaluate_stem(
        arguments.root_dir,
        arguments.checkpoint_dir,
        arguments.config,
        arguments.stem,
        arguments.checkpoint_path,
    )
