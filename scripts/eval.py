"""CLI evaluation script for remote sensing dehazing models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.datasets.dataset import RemoteSensingDehazeDataset
from src.eval.evaluator import Evaluator
from src.models.aodnet import AODNet
from src.models.dcp import RSDCP
from src.models.rs_pgnet import RSPGNet
from src.utils.config import load_config
from src.utils.seed import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a remote sensing dehazing model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="rs_pgnet", choices=["rs_dcp", "aodnet", "rs_pgnet"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint.")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples to evaluate.")
    parser.add_argument("--output-dir", type=str, default="experiments/eval_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.project.seed)
    device = get_device(config.hardware.device)

    # Initialize model
    if args.model == "rs_dcp":
        model = RSDCP()
    elif args.model == "aodnet":
        model = AODNet(in_channels=config.model.in_channels)
    elif args.model == "rs_pgnet":
        model = RSPGNet(
            in_channels=config.model.in_channels,
            base_channels=config.model.base_channels,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {args.checkpoint}")

    # Create dataset
    test_ds = RemoteSensingDehazeDataset(
        root_dir=config.data.data_dir,
        split="test",
        image_size=config.data.image_size,
        num_synthetic_samples=args.samples,
        seed=config.project.seed + 9999,
        augment=False,
    )

    evaluator = Evaluator(model=model, device=device, output_dir=args.output_dir)
    metrics = evaluator.evaluate_dataset(test_ds, save_images=True)
    print(f"\nFinal Test Results [{args.model}]:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
