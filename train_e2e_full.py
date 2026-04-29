#!/usr/bin/env python3
"""
Full Pipeline: End-to-End NN Control with Imitation + RL

Stages:
1. Collect PD controller data from Stage 1
2. Imitation learning to pretrain NN
3. RL fine-tuning with PPO on full curriculum

Usage:
    python train_e2e_full.py --episodes 1000 --epochs 50 --steps 2000000 --seeds 0

This script runs the complete pipeline.
"""

import os
import sys
import argparse
import subprocess

# Import our modules
from collect_pd_data import collect_pd_data
from imitation_pretrain import main as imitation_main


def run_command(cmd, description):
    """Run a command and print description."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end NN control pipeline: data collection → imitation → RL"
    )
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Episodes for PD data collection")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs for imitation learning")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Total RL training steps")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seed values for RL")
    parser.add_argument("--skip-data-collection", action="store_true",
                        help="Skip data collection (use existing data)")
    parser.add_argument("--skip-imitation", action="store_true",
                        help="Skip imitation learning (use existing model)")
    parser.add_argument("--data-path", type=str, default="data/pd_stage1_buffer.npz",
                        help="Path to PD data")
    parser.add_argument("--pretrained-path", type=str, default="models/nn_pretrained_best.pt",
                        help="Path to save pretrained model")

    args = parser.parse_args()

    # Step 1: Collect PD data
    if not args.skip_data_collection:
        print(f"\n{'#'*60}")
        print("# STEP 1: Collecting PD data from Stage 1")
        print(f"{'#'*60}")
        collect_pd_data(
            n_episodes=args.episodes,
            save_path=args.data_path,
            stage=1,
        )
    else:
        print("Skipping data collection (using existing data)")

    # Step 2: Imitation learning
    if not args.skip_imitation:
        print(f"\n{'#'*60}")
        print("# STEP 2: Imitation Learning (PD → NN)")
        print(f"{'#'*60}")

        # Parse data path for loading
        import argparse as ap
        from pathlib import Path

        # Build arguments for imitation_pretrain
        imitation_args = [
            "python", "imitation_pretrain.py",
            "--data", args.data_path,
            "--epochs", str(args.epochs),
            "--save", args.pretrained_path,
        ]

        # Override with command line args
        sys.argv = imitation_args
        imitation_main()
    else:
        print("Skipping imitation learning (using existing model)")

    # Step 3: RL fine-tuning
    print(f"\n{'#'*60}")
    print("# STEP 3: RL Fine-tuning with PPO")
    print(f"{'#'*60}")

    e2e_args = [
        "python", "train_e2e_ppo.py",
        "--pretrained", args.pretrained_path,
        "--seeds", args.seeds,
        "--steps", str(args.steps),
    ]

    result = subprocess.run(e2e_args)
    if result.returncode != 0:
        print("ERROR: RL training failed!")
        return

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Data: {args.data_path}")
    print(f"  Pretrained: {args.pretrained_path}")
    print(f"  Seeds: {args.seeds}")


if __name__ == "__main__":
    main()