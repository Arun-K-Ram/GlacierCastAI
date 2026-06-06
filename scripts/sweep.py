"""
W&B hyperparameter sweep launcher for GlacierCastAI.

Usage:
    # Launch a new sweep
    python scripts/sweep.py --config configs/sweep.yaml --count 60

    # Resume an existing sweep
    python scripts/sweep.py --sweep-id <id> --count 30

Sweep strategy: Bayesian optimization (not random/grid).
Bayesian uses results of previous runs to pick next hyperparameters -
much more sample-efficient than grid search for 7+ dimensions.

Search space (configs/sweep.yaml):
    - learning rate
    - batch size
    - hidden dim
    - num layers
    - sequence length
    - backbone
    - temporal model type
    - loss weights
    - freeze epochs
"""

import argparse
import logging

import wandb
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch GlacierCastAI W&B sweep")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--project", type=str, default="GlacierCastAI")
    parser.add_argument("--sweep-id", type=str, default=None)
    return parser.parse_args()


def run_agent(sweep_id: str, project: str, count: int) -> None:
    """Run sweep agent - calls train.py for each hyperparameter config."""
    import subprocess
    import sys

    def train_fn():
        # W&B initializes a run and injects config via wandb.config
        run = wandb.init()
        cfg = dict(run.config)

        # Build CLI args from sweep config
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", "configs/model.yaml",
        ]

        # Map sweep params to CLI args
        param_map = {
            "training.optimizer.lr":              "--lr",
            "training.batch_size":                "--batch-size",
            "model.temporal.hidden_dim":          "--hidden-dim",
            "data.sequences.length":              "--seq-len",
            "model.backbone.type":                "--backbone",
            "model.temporal.type":                "--temporal",
            "model.backbone.freeze_epochs":       "--freeze-epochs",
            "model.loss_weights.retreat_rate":    "--retreat-weight",
            "model.loss_weights.risk_score":      "--risk-weight",
        }

        for param_key, cli_flag in param_map.items():
            # Handle nested keys
            keys = param_key.split(".")
            val = cfg
            for k in keys:
                val = val.get(k, None)
                if val is None:
                    break
            if val is not None:
                cmd.extend([cli_flag, str(val)])

        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    wandb.agent(sweep_id, function=train_fn, count=count, project=project)


def main():
    args = parse_args()

    if args.sweep_id:
        # Resume existing sweep
        logger.info(f"Resuming sweep: {args.sweep_id}")
        run_agent(args.sweep_id, args.project, args.count)
    else:
        # Create new sweep
        with open(args.config) as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.project)

        logger.info(f"Sweep created: {sweep_id}")
        logger.info(
            f"View at: https://wandb.ai/{args.project}/sweeps/{sweep_id}"
        )
        logger.info(f"Running {args.count} sweep trials...")

        run_agent(sweep_id, args.project, args.count)


if __name__ == "__main__":
    main()