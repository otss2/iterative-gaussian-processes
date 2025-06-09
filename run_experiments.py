#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse


def find_yaml_configs(base_path):
    """Find all yaml config files in the given path recursively and sort them by dataset, split, and variant."""
    configs = []
    base_path = Path(base_path)

    # Read dataset order from order.txt
    order_file = base_path / "order.txt"
    if not order_file.exists():
        raise FileNotFoundError(f"order.txt not found in {base_path}")

    dataset_order = order_file.read_text().strip().split("\n")

    # For each dataset in order
    for dataset in dataset_order:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            continue

        # Get all splits across vanilla and sym
        all_splits = set()
        for variant in ["vanilla", "sym"]:
            variant_path = dataset_path / variant
            if not variant_path.exists():
                continue
            for file in variant_path.glob("split*.yaml"):
                split_num = int(file.stem.replace("split", ""))
                all_splits.add(split_num)

        # For each split number
        for split_num in sorted(all_splits):
            # First vanilla, then sym
            for variant in ["vanilla", "sym"]:
                config_path = dataset_path / variant / f"split{split_num}.yaml"
                if config_path.exists():
                    # Get path relative to base_path and remove .yaml extension
                    relative_path = config_path.relative_to(base_path)
                    config_name = str(relative_path.with_suffix(""))
                    configs.append(config_name)

    return configs  # Already sorted by our custom logic


def main():
    parser = argparse.ArgumentParser(description="Run Hydra experiments from a specified folder.")
    parser.add_argument("folder", type=str, help="Parent folder within conf/ to search for experiments (e.g., exp1)")
    args = parser.parse_args()

    # Path to your experiment configs
    exp_path = Path("conf") / args.folder

    if not exp_path.exists():
        print(f"Error: {exp_path} does not exist!")
        return

    # Find all yaml configs (excluding base configs)
    configs = find_yaml_configs(exp_path)

    if not configs:
        print(f"No yaml configs found in {exp_path}")
        return

    print(f"\nFound {len(configs)} configurations in {args.folder}:")
    for config in configs:
        print(f"  - {config}")

    # Create the multirun command using e.g. +exp1= to append configs
    override_str = f"'+{args.folder}={','.join(configs)}'"

    print("\nRunning command:")
    cmd = f"python train.py --multirun {override_str}"
    print(cmd)
    print("\nStarting experiments...\n")

    # Run the command
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
