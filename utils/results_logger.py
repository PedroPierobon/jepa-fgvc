"""
Experiment Logging and Results Management for FEPA-FGVC
======================================================
Provides structured saving for experiment configurations, performance metrics,
human-readable evaluation reports, and automatic row logging into a master
benchmark summary CSV (results/benchmark_summary.csv).
"""

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit_hash() -> str:
    """Returns current git commit hash if running in a git repository."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_experiment_result(
    experiment_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    report_text: Optional[str] = None,
    output_root: str = "results",
) -> Path:
    """
    Saves complete experiment results in an organized folder structure:
    - results/<experiment_name>_<YYYYMMDD_HHMMSS>/config.json
    - results/<experiment_name>_<YYYYMMDD_HHMMSS>/metrics.json
    - results/<experiment_name>_<YYYYMMDD_HHMMSS>/report.txt
    - Appends entry into results/benchmark_summary.csv

    Args:
        experiment_name: Identifier for the experiment (e.g. 'eval_swin_t_fold0', 'ft_efficientnetv2')
        config: Dictionary or argparse Namespace containing all configuration parameters.
        metrics: Dictionary of evaluated metrics (accuracies, invalid rates, losses).
        report_text: Optional formatted string table of the evaluation report.
        output_root: Base results directory (default: 'results').

    Returns:
        Path to the newly created experiment folder.
    """
    results_dir = Path(output_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_exp_name = experiment_name.replace("/", "_").replace(" ", "_")
    exp_folder_name = f"{sanitized_exp_name}_{timestamp_str}"
    exp_dir = results_dir / exp_folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to dictionary if passed as Namespace
    if hasattr(config, "__dict__"):
        config_dict = vars(config).copy()
    else:
        config_dict = dict(config).copy()

    # Sanitize config values (convert non-serializable objects to string)
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            config_dict[k] = str(v)

    config_dict["_timestamp"] = timestamp_str
    config_dict["_git_commit"] = get_git_commit_hash()
    config_dict["_experiment_folder"] = str(exp_dir)

    # 1. Save config.json
    config_file = exp_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 2. Save metrics.json
    # Convert any PyTorch tensors or numpy types in metrics
    clean_metrics: Dict[str, Any] = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            clean_metrics[k] = v.item()
        elif hasattr(v, "tolist"):
            clean_metrics[k] = v.tolist()
        elif isinstance(v, dict):
            clean_metrics[k] = {
                sub_k: (sub_v.item() if hasattr(sub_v, "item") else sub_v)
                for sub_k, sub_v in v.items()
            }
        else:
            clean_metrics[k] = v

    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)

    # 3. Save report.txt
    if report_text:
        report_file = exp_dir / "report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

    # 4. Append row to results/benchmark_summary.csv
    csv_file = results_dir / "benchmark_summary.csv"
    csv_headers = [
        "timestamp",
        "experiment_name",
        "backbone",
        "split_fold",
        "epochs",
        "marginal_acc",
        "type_acc",
        "make_acc",
        "model_acc",
        "exact_tuple_acc",
        "invalid_tuples_pct",
        "invalid_make_model_pct",
        "invalid_model_type_pct",
        "rgb_exact_acc",
        "ir_exact_acc",
        "folder",
    ]

    file_exists = csv_file.exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if not file_exists:
            writer.writeheader()

        row = {
            "timestamp": timestamp_str,
            "experiment_name": sanitized_exp_name,
            "backbone": config_dict.get("backbone", "N/A"),
            "split_fold": config_dict.get("split_fold", "N/A"),
            "epochs": config_dict.get("epochs", "N/A"),
            "marginal_acc": f"{clean_metrics.get('marginal_acc', 0.0):.2f}",
            "type_acc": f"{clean_metrics.get('acc_type', 0.0):.2f}",
            "make_acc": f"{clean_metrics.get('acc_make', 0.0):.2f}",
            "model_acc": f"{clean_metrics.get('acc_model', 0.0):.2f}",
            "exact_tuple_acc": f"{clean_metrics.get('acc_exact_match', 0.0):.2f}",
            "invalid_tuples_pct": f"{clean_metrics.get('pct_invalid_total', 0.0):.2f}",
            "invalid_make_model_pct": f"{clean_metrics.get('pct_invalid_make_model', 0.0):.2f}",
            "invalid_model_type_pct": f"{clean_metrics.get('pct_invalid_model_type', 0.0):.2f}",
            "rgb_exact_acc": f"{clean_metrics.get('rgb', {}).get('acc_exact', 0.0):.2f}",
            "ir_exact_acc": f"{clean_metrics.get('ir', {}).get('acc_exact', 0.0):.2f}",
            "folder": str(exp_dir),
        }
        writer.writerow(row)

    print(f"[✓] Experiment results saved to: {exp_dir}")
    print(f"[✓] Master summary updated in:  {csv_file}")
    return exp_dir
