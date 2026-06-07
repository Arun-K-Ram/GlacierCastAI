"""
Evaluation script for GlacierCastAI.

Loads best checkpoint, runs inference on test set,
computes all metrics, generates paper figures and tables.

Usage:
    python scripts/evaluate.py --config configs/model.yaml --checkpoint experiments/checkpoints/best.ckpt
    python scripts/evaluate.py --config configs/model.yaml --checkpoint experiments/checkpoints/best.ckpt --explain
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb

from src.training.trainer import GlacierCastAIModule
from src.training.datamodule import GlacierDataModule
from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.explainability.attribution import explain_glacier, generate_driver_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GlacierCastAI")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="paper")
    parser.add_argument("--explain",    action="store_true",
                        help="Run SHAP + GradCAM++ explainability")
    parser.add_argument("--device",     type=str, default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict) -> GlacierCastAIModule:
    model = GlacierCastAIModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
    )
    model.eval()
    return model


def run_evaluation(
    model: GlacierCastAIModule,
    datamodule: GlacierDataModule,
    device: str,
) -> list[dict]:
    """
    Run inference on test set and collect per-sample metrics.

    Returns:
        List of per-sample metric dicts.
    """
    model = model.to(device)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_metrics = []

    with torch.no_grad():
        for batch in test_loader:
            image_seq   = batch["image_seq"].to(device)
            climate_seq = batch["climate_seq"].to(device)
            dem         = batch["dem"].to(device)

            outputs = model(image_seq, climate_seq, dem)

            pred_mask    = torch.sigmoid(outputs["mask"]).squeeze(1).cpu().numpy()
            target_mask  = batch["target_mask"].squeeze(1).numpy()
            pred_retreat = outputs["retreat"].cpu().numpy()
            tgt_retreat  = batch["target_retreat"].numpy()

            for i in range(pred_mask.shape[0]):
                metrics = compute_all_metrics(
                    pred_mask=pred_mask[i],
                    target_mask=target_mask[i],
                    pred_retreat=pred_retreat[i],
                    target_retreat=tgt_retreat[i],
                )
                all_metrics.append(metrics)

    return all_metrics


def print_results_table(aggregated: dict) -> None:
    """Print results in paper-ready format."""
    print("\n" + "=" * 60)
    print("GLACIERCASTAI - TEST SET RESULTS")
    print("=" * 60)

    metrics = [
        ("IoU",           "iou_mean",           "iou_std"),
        ("Boundary F1",   "boundary_f1_mean",   "boundary_f1_std"),
        ("Retreat MAE",   "retreat_mae_mean",   "retreat_mae_std"),
        ("Retreat RMSE",  "retreat_rmse_mean",  "retreat_rmse_std"),
        ("Area Error km²","area_error_km2_mean","area_error_km2_std"),
    ]

    for name, mean_key, std_key in metrics:
        mean = aggregated.get(mean_key, 0)
        std  = aggregated.get(std_key, 0)
        print(f"  {name:<20}: {mean:.4f} ± {std:.4f}")

    print("=" * 60 + "\n")


def save_latex_table(aggregated: dict, output_path: Path) -> None:
    """
    Save results as LaTeX table for paper.
    Paste directly into paper/tables/table_results.tex
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{GlacierCastAI test set evaluation results}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Metric & Mean & Std \\",
        r"\hline",
    ]

    metrics = [
        ("IoU",            "iou_mean",            "iou_std"),
        ("Boundary F1",    "boundary_f1_mean",    "boundary_f1_std"),
        ("Retreat MAE",    "retreat_mae_mean",    "retreat_mae_std"),
        ("Retreat RMSE",   "retreat_rmse_mean",   "retreat_rmse_std"),
        ("Area Error km²", "area_error_km2_mean", "area_error_km2_std"),
    ]

    for name, mean_key, std_key in metrics:
        mean = aggregated.get(mean_key, 0)
        std  = aggregated.get(std_key, 0)
        lines.append(f"{name} & {mean:.4f} & {std:.4f} \\\\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"LaTeX table saved: {output_path}")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    device     = args.device if torch.cuda.is_available() else "cpu"

    #  Load model 
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config)

    #  Load data 
    data_config = {
        **config["data"],
        "batch_size": config["training"]["batch_size"],
    }
    datamodule = GlacierDataModule(data_config)

    #  Run evaluation 
    logger.info("Running evaluation on test set...")
    all_metrics = run_evaluation(model, datamodule, device)

    aggregated = aggregate_metrics(all_metrics)

    #  Print + save results 
    print_results_table(aggregated)

    save_latex_table(
        aggregated,
        output_dir / "tables" / "table_results.tex",
    )

    #  Log to W&B 
    wandb.init(
        project="GlacierCastAI",
        job_type="evaluation",
        config=config,
    )
    wandb.log({f"test/{k}": v for k, v in aggregated.items()})
    wandb.finish()

    #  Explainability 
    if args.explain:
        logger.info("Running explainability analysis...")
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        sample = next(iter(test_loader))

        # Use training data mean as SHAP background
        background = np.zeros((50, len(config.get("climate_features", 16))))

        glaciers = list(config["data"]["glaciers"].keys())

        for glacier in glaciers:
            logger.info(f"Explaining: {glacier}")
            result = explain_glacier(
                model=model.model,
                sample={k: v[0] for k, v in sample.items()},
                background_data=background,
                glacier_name=glacier,
                device=device,
            )
            print(result["driver_report"])

            report_path = output_dir / "figures" / f"shap_{glacier}.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result["driver_report"])
            logger.info(f"Driver report saved: {report_path}")


if __name__ == "__main__":
    main()