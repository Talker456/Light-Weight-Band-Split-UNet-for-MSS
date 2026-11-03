import argparse

from evaluate_stem import STEMS, evaluate_stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb_path", required=True)
    parser.add_argument(
        "--config",
        default="configs/v5_proposed_s_scnet_seg9_mb1_ga4_half_epoch.yaml",
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--stems", nargs="+", default=["all"])
    parser.add_argument("--extension", default="flac")
    arguments = parser.parse_args()
    stems = STEMS if "all" in arguments.stems else arguments.stems
    for stem in stems:
        evaluate_stem(
            arguments.musdb_path,
            arguments.checkpoint_dir,
            arguments.config,
            stem,
        )
