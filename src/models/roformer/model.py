import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.roformer.blocks import (
    Bottleneck,
    LightRoformerAsymmetricDecoder,
    LightRoformerEncoder,
    RNNBottleneck,
)


class LightRoformerFinalStage(nn.Module):
    """K=3 band-wise projection followed by inverse equal-band rearrangement."""

    def __init__(self, channels, out_real_channels, n_band):
        super().__init__()
        if channels % n_band:
            raise ValueError("Final-stage channels must be divisible by n_band")
        self.out_real_channels = out_real_channels
        self.n_band = n_band
        self.projection = nn.Conv2d(
            channels,
            out_real_channels * n_band,
            kernel_size=3,
            padding=1,
            groups=n_band,
        )

    def forward(self, x):
        x = self.projection(x)
        batch, _, band_width, frames = x.shape
        x = x.reshape(
            batch, self.n_band, self.out_real_channels, band_width, frames
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(
            batch, self.out_real_channels, self.n_band * band_width, frames
        )


class LightRoformer(nn.Module):
    """Paper-aligned Moises-Light with reference decoder and scale restoration."""

    architecture_version = "v5"
    compatible_architecture_versions = frozenset({"v5", "paper_v5"})

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        n_band=4,
        G=32,
        n_layers=6,
        n_heads=8,
        bottleneck_type="attention",
        freq_dim=2048,
        n_enc=3,
        n_dec=1,
        n_split_enc=3,
        n_split_dec=1,
        bn_factor=8,
        dropout=0.1,
        gradient_checkpointing=False,
        norm_name="group",
        activation_name="relu",
        input_normalization="sample_zscore",
        output_mode="direct",
    ):
        super().__init__()
        if freq_dim % n_band:
            raise ValueError("freq_dim must be divisible by n_band")
        if G % n_band:
            raise ValueError("G is the total first-stage width and must be divisible by n_band")
        if bottleneck_type not in {"attention", "rnn"}:
            raise ValueError("bottleneck_type must be 'attention' or 'rnn'")
        if input_normalization not in {"sample_zscore", "none"}:
            raise ValueError("input_normalization must be 'sample_zscore' or 'none'")
        if output_mode not in {"direct", "mask", "legacy_direct"}:
            raise ValueError("output_mode must be 'direct', 'mask', or 'legacy_direct'")
        if output_mode == "mask" and in_channels != out_channels:
            raise ValueError("mask output requires matching input and output channels")

        self.n_band = n_band
        self.G = G
        self.freq_dim = freq_dim
        self.n_enc = n_enc
        self.out_channels = out_channels
        self.bottleneck_type = bottleneck_type
        self.norm_name = norm_name
        self.activation_name = activation_name
        self.input_normalization = input_normalization
        self.output_mode = output_mode
        self.n_dec = n_dec
        self.n_split_enc = n_split_enc
        self.n_split_dec = n_split_dec

        band_freq_dim = freq_dim // n_band
        self.encoder = LightRoformerEncoder(
            real_channels=in_channels * 2,
            n_band=n_band,
            channels=G,
            n_enc=n_enc,
            n_split=n_split_enc,
            freq_dim=band_freq_dim,
            bottleneck_factor=bn_factor,
            norm_name=norm_name,
            activation_name=activation_name,
        )
        bottleneck_channels = G * (n_enc + 1)
        if bottleneck_type == "attention":
            if bottleneck_channels % n_heads:
                raise ValueError("Bottleneck channels must be divisible by n_heads")
            self.bottleneck = Bottleneck(
                channels=bottleneck_channels,
                n_band=n_band,
                freq_dim=band_freq_dim,
                n_split=n_split_enc,
                num_layers=n_layers,
                num_heads=n_heads,
                dropout=dropout,
                bottleneck_factor=bn_factor,
                gradient_checkpointing=gradient_checkpointing,
                norm_name=norm_name,
                activation_name=activation_name,
            )
        else:
            self.bottleneck = RNNBottleneck(
                channels=bottleneck_channels,
                n_band=n_band,
                freq_dim=band_freq_dim,
                n_split=n_split_enc,
                num_layers=n_layers,
                bottleneck_factor=bn_factor,
                norm_name=norm_name,
                activation_name=activation_name,
            )
        self.decoder = LightRoformerAsymmetricDecoder(
            channels=G,
            n_band=n_band,
            n_enc=n_enc,
            n_dec=n_dec,
            n_split=n_split_dec,
            freq_dim=band_freq_dim,
            bottleneck_factor=bn_factor,
            norm_name=norm_name,
            activation_name=activation_name,
        )
        self.final_stage = LightRoformerFinalStage(
            channels=G,
            out_real_channels=out_channels * 2,
            n_band=n_band,
        )

    def forward(self, x):
        if not torch.is_complex(x):
            raise TypeError("LightRoformer expects a complex spectrogram")
        _, _, original_freqs, original_frames = x.shape

        # DTTNet/Moises-Light process the first 2048 bins and zero-pad for iSTFT.
        if original_freqs < self.freq_dim:
            truncated = x
        else:
            truncated = x[:, :, : self.freq_dim]

        # Match the reference implementation: normalize real/imag features per
        # sample, then restore the mixture scale for direct spectrogram output.
        original_features = torch.cat([truncated.real, truncated.imag], dim=1).float()
        if self.input_normalization == "sample_zscore":
            mean = original_features.mean(dim=(1, 2, 3), keepdim=True)
            std = original_features.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            model_input = (original_features - mean) / std
        else:
            mean = original_features.new_zeros((original_features.shape[0], 1, 1, 1))
            std = original_features.new_ones((original_features.shape[0], 1, 1, 1))
            model_input = original_features

        if original_freqs < self.freq_dim:
            model_input = F.pad(
                model_input, (0, 0, 0, self.freq_dim - original_freqs)
            )
            mask_features = F.pad(
                original_features, (0, 0, 0, self.freq_dim - original_freqs)
            )
        else:
            mask_features = original_features

        time_multiple = 2 ** self.n_enc
        pad_frames = (-original_frames) % time_multiple
        if pad_frames:
            model_input = F.pad(model_input, (0, pad_frames))

        latent, skips = self.encoder(model_input)
        latent = self.bottleneck(latent)
        decoded = self.decoder(latent, skips)
        out = self.final_stage(decoded)[..., :original_frames]

        if self.output_mode == "direct":
            out = out * std + mean
        elif self.output_mode == "mask":
            out = out * mask_features

        real, imag = out.chunk(2, dim=1)
        out = torch.complex(real.float(), imag.float())
        if original_freqs > self.freq_dim:
            out = F.pad(out, (0, 0, 0, original_freqs - self.freq_dim))
        else:
            out = out[:, :, :original_freqs]
        return out

    def architecture_metadata(self):
        return {
            "architecture_version": self.architecture_version,
            "bottleneck_type": self.bottleneck_type,
            "freq_dim": self.freq_dim,
            "n_band": self.n_band,
            "G": self.G,
            "n_enc": self.n_enc,
            "n_dec": self.n_dec,
            "n_split_enc": self.n_split_enc,
            "n_split_dec": self.n_split_dec,
            "normalization": self.norm_name,
            "activation": self.activation_name,
            "input_normalization": self.input_normalization,
            "output_mode": self.output_mode,
        }

    def validate_checkpoint_metadata(self, metadata):
        expected = self.architecture_metadata()
        mismatches = {}
        for key, value in expected.items():
            checkpoint_value = metadata.get(key)
            if (
                key == "architecture_version"
                and checkpoint_value in self.compatible_architecture_versions
            ):
                continue
            if checkpoint_value != value:
                mismatches[key] = {
                    "checkpoint": checkpoint_value,
                    "expected": value,
                }
        if mismatches:
            details = ", ".join(
                f"{key}={values['checkpoint']!r} (expected {values['expected']!r})"
                for key, values in mismatches.items()
            )
            raise RuntimeError(f"Incompatible checkpoint metadata: {details}")
