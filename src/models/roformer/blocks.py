import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.roformer.attention import InterleavedRoPEBlock
from src.models.roformer.dual_path import DualPathRNN
from src.models.roformer.layers import (
    LightRoformerInitialStage,
    SplitModule,
    SplitAndMergeModule,
    TimeDownsample,
    TimeUpsample,
    make_norm,
)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_band,
        n_split,
        freq_dim,
        bottleneck_factor,
        norm_name,
        activation_name,
    ):
        super().__init__()
        self.smm = SplitAndMergeModule(
            channels,
            n_band,
            n_split,
            freq_dim,
            bottleneck_factor,
            norm_name,
            activation_name,
        )
        self.downsample = TimeDownsample(
            channels, out_channels, norm_name, activation_name
        )

    def forward(self, x):
        skip = self.smm(x)
        return self.downsample(skip), skip


class LightRoformerEncoder(nn.Module):
    def __init__(
        self,
        real_channels=4,
        n_band=4,
        channels=32,
        n_enc=3,
        n_split=3,
        freq_dim=512,
        bottleneck_factor=8,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.initial_stage = LightRoformerInitialStage(
            real_channels, n_band, channels, norm_name, activation_name
        )
        self.blocks = nn.ModuleList()
        current = channels
        for _ in range(n_enc):
            next_channels = current + channels
            self.blocks.append(
                EncoderBlock(
                    current,
                    next_channels,
                    n_band,
                    n_split,
                    freq_dim,
                    bottleneck_factor,
                    norm_name,
                    activation_name,
                )
            )
            current = next_channels
        self.out_channels = current

    def forward(self, x):
        x = self.initial_stage(x)
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        return x, skips


class Bottleneck(nn.Module):
    def __init__(
        self,
        channels,
        n_band,
        freq_dim,
        n_split=3,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        bottleneck_factor=8,
        gradient_checkpointing=False,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_split = SplitAndMergeModule(
            channels,
            n_band,
            n_split,
            freq_dim,
            bottleneck_factor,
            norm_name,
            activation_name,
        )
        self.layers = nn.ModuleList(
            [InterleavedRoPEBlock(channels, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        # Pre-norm transformer residuals can accumulate scale across the time and
        # frequency paths.  Bound the distribution before it reaches the
        # multiplicative decoder fusion.
        self.output_norm = make_norm(channels, norm_name)

    def forward(self, x):
        x = self.pre_split(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.output_norm(x)


class RNNBottleneck(nn.Module):
    """Legacy compatibility path; paper presets use the RoPE bottleneck."""

    def __init__(
        self,
        channels,
        n_band,
        freq_dim,
        n_split=3,
        num_layers=4,
        bottleneck_factor=8,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.pre_split = SplitAndMergeModule(
            channels,
            n_band,
            n_split,
            freq_dim,
            bottleneck_factor,
            norm_name,
            activation_name,
        )
        self.layers = nn.ModuleList([DualPathRNN(channels) for _ in range(num_layers)])

    def forward(self, x):
        x = self.pre_split(x)
        for layer in self.layers:
            x = layer(x)
        return x


class LightRoformerAsymmetricDecoder(nn.Module):
    """Heavy early refinement with stable lightweight refinement at every stage."""

    def __init__(
        self,
        channels,
        n_band,
        n_enc,
        n_dec,
        n_split,
        freq_dim,
        bottleneck_factor,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        if not 0 <= n_dec <= n_enc:
            raise ValueError("n_dec must be between zero and n_enc")
        self.upsamples = nn.ModuleList()
        self.fusion_norms = nn.ModuleList()
        self.refine = nn.ModuleList()
        self.capture_diagnostics = False
        self.last_diagnostics = []
        current = channels * (n_enc + 1)
        for stage in range(n_enc):
            out_channels = current - channels
            self.upsamples.append(
                TimeUpsample(current, out_channels, norm_name, activation_name)
            )
            # The DTTNet decoder refines every multiplicative skip fusion.  A
            # per-sample-safe normalization here also keeps FP16 convolutions
            # from seeing the rare, very large product outliers observed in v4.
            self.fusion_norms.append(make_norm(out_channels, norm_name))
            self.refine.append(
                SplitAndMergeModule(
                    out_channels,
                    n_band,
                    n_split,
                    freq_dim,
                    bottleneck_factor,
                    norm_name,
                    activation_name,
                )
                if stage < n_dec
                else SplitModule(
                    out_channels,
                    out_channels,
                    n_band,
                    kernel_size=3,
                    norm_name=norm_name,
                    activation_name=activation_name,
                )
            )
            current = out_channels

    @staticmethod
    def _tensor_stats(x):
        values = x.detach().float().flatten()
        finite = torch.isfinite(values)
        nonfinite = int((~finite).sum().item())
        values = values[finite]
        if values.numel() == 0:
            return {"nonfinite": nonfinite}
        stride = max(1, values.numel() // 65536)
        sample = values[::stride]
        absolute = sample.abs()
        quantiles = torch.quantile(
            absolute, sample.new_tensor([0.5, 0.99, 0.999])
        )
        rms = values.square().mean().sqrt()
        return {
            "rms": float(rms.cpu()),
            "abs_p50": float(quantiles[0].cpu()),
            "abs_p99": float(quantiles[1].cpu()),
            "abs_p999": float(quantiles[2].cpu()),
            "abs_max": float(absolute.max().cpu()),
            "positive_fraction": float((sample > 0).float().mean().cpu()),
            "near_zero_fraction": float((absolute < 1e-4).float().mean().cpu()),
            "nonfinite": nonfinite,
        }

    def forward(self, x, skips):
        if len(skips) != len(self.upsamples):
            raise ValueError(f"Expected {len(self.upsamples)} skip tensors, got {len(skips)}")
        diagnostics = []
        for stage, (upsample, fusion_norm, refine, skip) in enumerate(
            zip(self.upsamples, self.fusion_norms, self.refine, reversed(skips))
        ):
            stage_diagnostics = None
            if self.capture_diagnostics:
                stage_diagnostics = {
                    "stage": stage,
                    "input": self._tensor_stats(x),
                    "skip": self._tensor_stats(skip),
                }
            if stage_diagnostics is not None:
                convolution = upsample.net[0](x)
                normalized = upsample.net[1](convolution)
                x = upsample.net[2](normalized)
                stage_diagnostics["convolution"] = self._tensor_stats(convolution)
                stage_diagnostics["normalization"] = self._tensor_stats(normalized)
                stage_diagnostics["activation"] = self._tensor_stats(x)
                stage_diagnostics["upsample"] = stage_diagnostics["activation"]
            else:
                x = upsample(x)
            if x.shape != skip.shape:
                raise RuntimeError(f"Decoder/skip shape mismatch: {x.shape} vs {skip.shape}")
            # Calculate the unbounded product in FP32.  The following
            # normalization returns it to a controlled range before any
            # autocast convolution can narrow it to FP16.
            product = x.float() * skip.float()
            fused = fusion_norm(product)
            if stage_diagnostics is not None:
                stage_diagnostics["product"] = self._tensor_stats(product)
                stage_diagnostics["fusion_normalization"] = self._tensor_stats(fused)
            x = refine(fused)
            if stage_diagnostics is not None:
                stage_diagnostics["output"] = self._tensor_stats(x)
                diagnostics.append(stage_diagnostics)
        self.last_diagnostics = diagnostics
        return x
