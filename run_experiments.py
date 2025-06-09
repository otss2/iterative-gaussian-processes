#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse


def find_yaml_configs(base_path):
    """Find all yaml config files in the given path recursively."""
    configs = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".yaml") and not file.startswith("_"):  # Skip base configs
                # Convert the path to be relative to the parent folder
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path)
                # Remove the .yaml extension
                config_path = os.path.splitext(relative_path)[0]
                configs.append(config_path)
    return sorted(configs)  # Sort for consistent ordering


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

    # Create the multirun command using +exp1= to append configs
    override_str = f"'+exp1={','.join(configs)}'"

    print("\nRunning command:")
    cmd = f"python train.py --multirun {override_str}"
    print(cmd)
    print("\nStarting experiments...\n")

    # Run the command
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
