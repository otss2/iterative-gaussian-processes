#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import shutil


def create_config_file(path: Path, dataset: str, split: int, symmetric: bool):
    """Create a YAML config file with the given parameters."""
    content = f"""# @package _global_

defaults:
  - /config

# Experiment-specific settings
dataset_name: {dataset}
dataset_split: {split}
use_symmetric_features: {str(symmetric).lower()}
"""
    path.write_text(content)


def cleanup_exp_dir(exp_path: Path):
    """Remove yaml configs and dataset directories, warn about other files."""
    if not exp_path.exists():
        return

    # Keep exp_detailts.txt, remove yaml files, warn about others
    for item in exp_path.iterdir():
        if item.name == "exp_detailts.txt":
            continue

        if item.is_file():
            if item.suffix == ".yaml":
                item.unlink()
            else:
                if item.name != "order.txt":
                    print(f"Warning: Found non-yaml file '{item}' - keeping it intact")
                else:
                    item.unlink()
        elif item.is_dir():
            # For directories, we'll check if they only contain yaml files
            has_non_yaml = False
            for file in item.rglob("*"):
                if file.is_file() and file.suffix != ".yaml":
                    has_non_yaml = True
                    print(f"Warning: Found non-yaml file '{file}' in dataset directory - keeping the directory intact")
                    break

            if not has_non_yaml:
                shutil.rmtree(item)

    print(f"Cleaned up yaml configs in {exp_path}")


def generate_configs(exp_dir: str):
    """Generate experiment configs based on the details file."""
    exp_path = Path("conf") / exp_dir
    details_path = exp_path / "exp_detailts.txt"

    if not details_path.exists():
        raise FileNotFoundError(f"Details file not found at {details_path}")

    # Clean up existing configs
    cleanup_exp_dir(exp_path)

    # Read and parse the details file
    datasets = []
    splits = {}
    with open(details_path) as f:
        for line in f:
            dataset, n_splits = line.strip().split()
            datasets.append(dataset)
            splits[dataset] = int(n_splits)

    # Write order.txt
    order_file = exp_path / "order.txt"
    order_file.write_text("\n".join(datasets))

    # Create directory structure and config files
    for dataset in datasets:
        dataset_dir = exp_path / dataset

        # Create vanilla and sym directories
        for variant in ["vanilla", "sym"]:
            variant_dir = dataset_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)

            # Create config files for each split
            for split in range(splits[dataset]):
                config_file = variant_dir / f"{split}.yaml"
                create_config_file(config_file, dataset=dataset, split=split, symmetric=(variant == "sym"))

    print(f"Generated experiment structure in {exp_path}")
    print(f"Created configs for datasets: {', '.join(datasets)}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configs from a details file.")
    parser.add_argument("exp_dir", type=str, help="Experiment directory name (e.g., exp2)")
    args = parser.parse_args()

    generate_configs(args.exp_dir)


if __name__ == "__main__":
    main()
