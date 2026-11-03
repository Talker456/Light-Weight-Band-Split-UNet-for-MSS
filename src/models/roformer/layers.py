import torch
import torch.nn as nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a GroupNorm group count that also works with batch size one."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(channels: int, norm_name: str = "batch") -> nn.Module:
    normalized_name = norm_name.lower().replace("_", "")
    if normalized_name in {"batch", "batchnorm", "batchnorm2d"}:
        return nn.BatchNorm2d(channels)
    if normalized_name in {"instance", "instancenorm", "instancenorm2d"}:
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized_name in {"group", "groupnorm"}:
        return nn.GroupNorm(_group_count(channels), channels)
    if normalized_name in {"layer", "layernorm"}:
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported normalization: {norm_name}")


def make_activation(activation_name: str = "relu") -> nn.Module:
    normalized_name = activation_name.lower().replace("_", "")
    if normalized_name == "relu":
        return nn.ReLU()
    if normalized_name == "gelu":
        return nn.GELU()
    if normalized_name in {"leakyrelu", "lrelu"}:
        return nn.LeakyReLU(0.01)
    if normalized_name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation_name}")


class SplitModule(nn.Module):
    """Band-wise convolution used by the paper's split modules."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_band,
        kernel_size=3,
        activate=True,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        if in_channels % n_band or out_channels % n_band:
            raise ValueError(
                f"SplitModule channels must be divisible by n_band={n_band}: "
                f"in={in_channels}, out={out_channels}"
            )
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=n_band,
            ),
            make_norm(out_channels, norm_name),
        ]
        if activate:
            layers.append(make_activation(activation_name))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrequencyTDF(nn.Module):
    """Time-distributed fully-connected bottleneck along frequency."""

    def __init__(
        self,
        channels,
        freq_dim,
        bottleneck_factor=8,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        hidden = max(1, freq_dim // bottleneck_factor)
        self.freq_dim = freq_dim
        self.fc1 = nn.Linear(freq_dim, hidden)
        self.norm1 = make_norm(channels, norm_name)
        self.fc2 = nn.Linear(hidden, freq_dim)
        self.norm2 = make_norm(channels, norm_name)
        self.activation = make_activation(activation_name)

    def forward(self, x):
        if x.shape[-2] != self.freq_dim:
            raise ValueError(
                f"TDF expected {self.freq_dim} frequency bins, got {x.shape[-2]}"
            )
        x = x.transpose(-1, -2).contiguous()  # B, C, T, F
        x = self.activation(self.norm1(self.fc1(x)))
        x = self.activation(self.norm2(self.fc2(x)))
        return x.transpose(-1, -2).contiguous()


class LightRoformerInitialStage(nn.Module):
    """Equal-width band split followed by the Enc-Dec V3 K=3 projection."""

    def __init__(
        self,
        real_channels,
        n_band,
        channels,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.real_channels = real_channels
        self.n_band = n_band
        self.projection = SplitModule(
            real_channels * n_band,
            channels,
            n_band,
            kernel_size=3,
            norm_name=norm_name,
            activation_name=activation_name,
        )

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=1)
        batch, channels, freqs, frames = x.shape
        if channels != self.real_channels:
            raise ValueError(f"Expected {self.real_channels} real channels, got {channels}")
        if freqs % self.n_band:
            raise ValueError(f"Frequency bins {freqs} are not divisible by {self.n_band}")
        band_width = freqs // self.n_band
        x = x.reshape(batch, channels, self.n_band, band_width, frames)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(batch, channels * self.n_band, band_width, frames)
        return self.projection(x)


class SplitAndMergeModule(nn.Module):
    """Enc-Dec V3 split/TDF/split block with an outer residual path."""

    def __init__(
        self,
        channels,
        n_band,
        n_split,
        freq_dim,
        bottleneck_factor=8,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        module_args = {
            "norm_name": norm_name,
            "activation_name": activation_name,
        }
        self.residual = SplitModule(
            channels, channels, n_band, kernel_size=3, **module_args
        )
        self.split1 = nn.Sequential(
            *[
                SplitModule(channels, channels, n_band, kernel_size=3, **module_args)
                for _ in range(n_split)
            ]
        )
        self.tdf = FrequencyTDF(
            channels,
            freq_dim,
            bottleneck_factor,
            norm_name=norm_name,
            activation_name=activation_name,
        )
        self.split2 = nn.Sequential(
            *[
                SplitModule(channels, channels, n_band, kernel_size=3, **module_args)
                for _ in range(n_split)
            ]
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.split1(x)
        x = x + self.tdf(x)
        x = self.split2(x)
        return x + residual


class TimeDownsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2)),
            make_norm(out_channels, norm_name),
            make_activation(activation_name),
        )

    def forward(self, x):
        return self.net(x)


class TimeUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="batch",
        activation_name="relu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2)),
            make_norm(out_channels, norm_name),
            make_activation(activation_name),
        )

    def forward(self, x):
        return self.net(x)
