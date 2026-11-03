from src.models.roformer.model import LightRoformer


def build_model(config, *, gradient_checkpointing=None):
    """Build the same architecture for training, evaluation, and inference."""
    model_config = config["model"]
    training_config = config.get("training", {})
    if gradient_checkpointing is None:
        gradient_checkpointing = training_config.get("gradient_checkpointing", False)
    model = LightRoformer(
        in_channels=model_config.get("in_channels", 2),
        out_channels=model_config.get("out_channels", 2),
        n_band=model_config.get("num_bands", 4),
        G=model_config.get("G", 32),
        n_layers=model_config.get("n_rope", 6),
        n_heads=model_config.get("num_heads", 8),
        bottleneck_type=model_config.get("bottleneck_type", "attention"),
        freq_dim=model_config.get("freq_dim", 2048),
        n_enc=model_config.get("num_layers", 3),
        n_dec=model_config.get("num_dec_layers", 1),
        n_split_enc=model_config.get("n_split_enc", 3),
        n_split_dec=model_config.get("n_split_dec", 1),
        bn_factor=model_config.get("bn_factor", 8),
        dropout=model_config.get("dropout", 0.1),
        gradient_checkpointing=gradient_checkpointing,
        norm_name=model_config.get("normalization", "group"),
        activation_name=model_config.get("activation", "relu"),
        input_normalization=model_config.get(
            "input_normalization", "sample_zscore"
        ),
        output_mode=model_config.get("output_mode", "direct"),
    )
    configured_version = model_config.get("architecture_version")
    if (
        configured_version
        and configured_version not in model.compatible_architecture_versions
    ):
        raise ValueError(
            f"Config requests architecture {configured_version!r}, but the code "
            f"builds {model.architecture_version!r}"
        )
    return model
