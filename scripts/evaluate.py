import argparse

from evaluate_stem import STEMS, evaluate_stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="data/musdb18hq_flac")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument(
        "--config",
        default="configs/v5_proposed_s_scnet_seg9_mb1_ga4_half_epoch.yaml",
    )
    arguments = parser.parse_args()
    for stem in STEMS:
        evaluate_stem(
            arguments.root_dir,
            arguments.checkpoint_dir,
            arguments.config,
            stem,
        )
