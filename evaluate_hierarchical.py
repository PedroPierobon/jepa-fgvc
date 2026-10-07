#!/usr/bin/env python3
"""
Hierarchical Evaluation & Taxonomic Consistency Assessment (UFPR-VeSV)
======================================================================
Trains a linear or multi-layer probe on frozen feature representations,
evaluates marginal classification accuracies (Type, Make, Model),
computes Exact-Match Full-Tuple Accuracy, and measures Invalid Tuple Rates
by validating against ground truth vehicle taxonomies.
Automatically logs results to results/<exp_name>/ and results/benchmark_summary.csv.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.results_logger import save_experiment_result


class VehicleTaxonomy:
    """
    Constructs and verifies the ground truth vehicle taxonomy of UFPR-VeSV.
    Identifies all valid (Type, Make, Model) combinations in the dataset.
    """

    def __init__(self, annotations_file: Path) -> None:
        self.annotations_file = Path(annotations_file)
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.unique_types = sorted(list(set(item["type"] for item in data)))
        self.unique_makes = sorted(list(set(item["make"] for item in data)))
        self.unique_models = sorted(list(set(item["model"] for item in data)))

        self.type_to_idx = {name: idx for idx, name in enumerate(self.unique_types)}
        self.make_to_idx = {name: idx for idx, name in enumerate(self.unique_makes)}
        self.model_to_idx = {name: idx for idx, name in enumerate(self.unique_models)}

        self.idx_to_type = {idx: name for name, idx in self.type_to_idx.items()}
        self.idx_to_make = {idx: name for name, idx in self.make_to_idx.items()}
        self.idx_to_model = {idx: name for name, idx in self.model_to_idx.items()}

        self.num_types = len(self.unique_types)
        self.num_makes = len(self.unique_makes)
        self.num_models = len(self.unique_models)
        self.cartesian_size = self.num_types * self.num_makes * self.num_models

        # Catalog valid tuples
        self.valid_tuples_idx: Set[Tuple[int, int, int]] = set()
        self.valid_make_model_idx: Set[Tuple[int, int]] = set()
        self.valid_model_type_idx: Set[Tuple[int, int]] = set()

        for item in data:
            t_idx = self.type_to_idx[item["type"]]
            m_idx = self.make_to_idx[item["make"]]
            mo_idx = self.model_to_idx[item["model"]]

            self.valid_tuples_idx.add((t_idx, m_idx, mo_idx))
            self.valid_make_model_idx.add((m_idx, mo_idx))
            self.valid_model_type_idx.add((mo_idx, t_idx))

        self.num_valid_tuples = len(self.valid_tuples_idx)
        self.valid_tuples_tensor = torch.tensor(list(self.valid_tuples_idx), dtype=torch.long)
        self.sorted_valid_tuples = sorted(list(self.valid_tuples_idx))
        self.tuple_to_idx = {t: idx for idx, t in enumerate(self.sorted_valid_tuples)}
        self.idx_to_tuple = {idx: t for idx, t in enumerate(self.sorted_valid_tuples)}

    def tuple_to_index(self, type_idx: int, make_idx: int, model_idx: int) -> int:
        return self.tuple_to_idx.get((type_idx, make_idx, model_idx), -1)

    def index_to_tuple(self, tuple_idx: int) -> Tuple[int, int, int]:
        return self.idx_to_tuple[tuple_idx]

    def is_valid_tuple(self, type_idx: int, make_idx: int, model_idx: int) -> bool:
        return (type_idx, make_idx, model_idx) in self.valid_tuples_idx

    def is_valid_make_model(self, make_idx: int, model_idx: int) -> bool:
        return (make_idx, model_idx) in self.valid_make_model_idx

    def is_valid_model_type(self, model_idx: int, type_idx: int) -> bool:
        return (model_idx, type_idx) in self.valid_model_type_idx


class HierarchicalMultiHeadProbe(nn.Module):
    """
    Multi-head linear or MLP probe evaluating Type (14), Make (26), and Model (136).
    """

    def __init__(
        self,
        embed_dim: int,
        num_types: int,
        num_makes: int,
        num_models: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.input_bn = nn.BatchNorm1d(embed_dim, affine=True)

        if hidden_dim > 0:
            self.shared_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = hidden_dim
        else:
            self.shared_mlp = nn.Identity()
            head_in = embed_dim

        self.head_type = nn.Linear(head_in, num_types)
        self.head_make = nn.Linear(head_in, num_makes)
        self.head_model = nn.Linear(head_in, num_models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_bn(x)
        feat = self.shared_mlp(x)
        return self.head_type(feat), self.head_make(feat), self.head_model(feat)


def decode_predictions_hcdc(
    probs_type: torch.Tensor,
    probs_make: torch.Tensor,
    probs_model: torch.Tensor,
    taxonomy: VehicleTaxonomy,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constrained Hierarchical Decoding (HCD-C):
    Projects probability distributions onto the valid taxonomic manifold V.
    """
    device = probs_type.device
    num_samples = probs_type.shape[0]

    log_p_type = torch.log(probs_type + 1e-12)
    log_p_make = torch.log(probs_make + 1e-12)
    log_p_model = torch.log(probs_model + 1e-12)

    valid_tuples = taxonomy.valid_tuples_tensor.to(device)  # [V, 3]
    t_indices = valid_tuples[:, 0]
    m_indices = valid_tuples[:, 1]
    mo_indices = valid_tuples[:, 2]

    scores_type = log_p_type[:, t_indices]    # [N, V]
    scores_make = log_p_make[:, m_indices]    # [N, V]
    scores_model = log_p_model[:, mo_indices] # [N, V]

    joint_scores = scores_type + scores_make + scores_model  # [N, V]
    best_tuple_indices = torch.argmax(joint_scores, dim=1)   # [N]

    best_tuples = valid_tuples[best_tuple_indices]  # [N, 3]
    return best_tuples[:, 0], best_tuples[:, 1], best_tuples[:, 2]


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_targets: Dict[str, torch.Tensor],
    val_embeddings: torch.Tensor,
    val_targets: Dict[str, torch.Tensor],
    taxonomy: VehicleTaxonomy,
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 128,
    hidden_dim: int = 0,
    device: torch.device = torch.device("cuda"),
) -> HierarchicalMultiHeadProbe:
    """Trains the linear or MLP evaluation probe on frozen features."""
    embed_dim = train_embeddings.shape[1]
    probe = HierarchicalMultiHeadProbe(
        embed_dim=embed_dim,
        num_types=taxonomy.num_types,
        num_makes=taxonomy.num_makes,
        num_models=taxonomy.num_models,
        hidden_dim=hidden_dim,
    ).to(device)

    train_ds = TensorDataset(
        train_embeddings,
        train_targets["type"],
        train_targets["make"],
        train_targets["model"],
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[*] Training Hierarchical Probe for {epochs} epochs (hidden_dim={hidden_dim})...")
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0

        for feats, y_t, y_m, y_mo in loader:
            feats = feats.to(device)
            y_t = y_t.to(device)
            y_m = y_m.to(device)
            y_mo = y_mo.to(device)

            optimizer.zero_grad()
            out_t, out_m, out_mo = probe(feats)
            loss = criterion(out_t, y_t) + criterion(out_m, y_m) + criterion(out_mo, y_mo)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                vf = val_embeddings.to(device)
                vt, vm, vmo = probe(vf)
                acc_t = (vt.argmax(dim=-1).cpu() == val_targets["type"]).float().mean().item() * 100
                acc_m = (vm.argmax(dim=-1).cpu() == val_targets["make"]).float().mean().item() * 100
                acc_mo = (vmo.argmax(dim=-1).cpu() == val_targets["model"]).float().mean().item() * 100
                print(
                    f"    Epoch [{epoch:02d}/{epochs:02d}] | Loss: {total_loss/len(loader):.4f} | "
                    f"Val Acc -> Type: {acc_t:.2f}%, Make: {acc_m:.2f}%, Model: {acc_mo:.2f}%"
                )

    return probe


def compute_hierarchical_metrics(
    pred_type: torch.Tensor,
    pred_make: torch.Tensor,
    pred_model: torch.Tensor,
    true_type: torch.Tensor,
    true_make: torch.Tensor,
    true_model: torch.Tensor,
    is_infrared: torch.Tensor,
    taxonomy: VehicleTaxonomy,
    prefix: str = "",
) -> Dict[str, Any]:
    """Computes all standard and fine-grained taxonomic consistency metrics."""
    pred_type_np = pred_type.cpu().numpy()
    pred_make_np = pred_make.cpu().numpy()
    pred_model_np = pred_model.cpu().numpy()

    true_type_np = true_type.cpu().numpy()
    true_make_np = true_make.cpu().numpy()
    true_model_np = true_model.cpu().numpy()
    is_ir_np = is_infrared.cpu().numpy().astype(bool)

    total_samples = len(true_type_np)

    correct_type = (pred_type_np == true_type_np)
    correct_make = (pred_make_np == true_make_np)
    correct_model = (pred_model_np == true_model_np)
    exact_match = correct_type & correct_make & correct_model

    acc_type = float(np.mean(correct_type) * 100.0)
    acc_make = float(np.mean(correct_make) * 100.0)
    acc_model = float(np.mean(correct_model) * 100.0)
    marginal_acc = float((acc_type + acc_make + acc_model) / 3.0)
    acc_exact_match = float(np.mean(exact_match) * 100.0)

    # Invalid tuple checks
    invalid_make_model = [not taxonomy.is_valid_make_model(int(pred_make_np[i]), int(pred_model_np[i])) for i in range(total_samples)]
    invalid_model_type = [not taxonomy.is_valid_model_type(int(pred_model_np[i]), int(pred_type_np[i])) for i in range(total_samples)]
    invalid_total = [not taxonomy.is_valid_tuple(int(pred_type_np[i]), int(pred_make_np[i]), int(pred_model_np[i])) for i in range(total_samples)]

    pct_invalid_make_model = float(np.mean(invalid_make_model) * 100.0)
    pct_invalid_model_type = float(np.mean(invalid_model_type) * 100.0)
    pct_invalid_total = float(np.mean(invalid_total) * 100.0)

    # RGB vs IR breakdown
    rgb_mask = ~is_ir_np
    ir_mask = is_ir_np

    metrics = {
        "prefix": prefix,
        "total_samples": total_samples,
        "acc_type": acc_type,
        "acc_make": acc_make,
        "acc_model": acc_model,
        "marginal_acc": marginal_acc,
        "acc_exact_match": acc_exact_match,
        "pct_invalid_make_model": pct_invalid_make_model,
        "pct_invalid_model_type": pct_invalid_model_type,
        "pct_invalid_total": pct_invalid_total,
        "rgb": {
            "count": int(np.sum(rgb_mask)),
            "acc_exact": float(np.mean(exact_match[rgb_mask]) * 100.0) if np.any(rgb_mask) else 0.0,
            "pct_invalid": float(np.mean([not taxonomy.is_valid_tuple(int(pred_type_np[i]), int(pred_make_np[i]), int(pred_model_np[i])) for i in np.where(rgb_mask)[0]]) * 100.0) if np.any(rgb_mask) else 0.0,
        },
        "ir": {
            "count": int(np.sum(ir_mask)),
            "acc_exact": float(np.mean(exact_match[ir_mask]) * 100.0) if np.any(ir_mask) else 0.0,
            "pct_invalid": float(np.mean([not taxonomy.is_valid_tuple(int(pred_type_np[i]), int(pred_make_np[i]), int(pred_model_np[i])) for i in np.where(ir_mask)[0]]) * 100.0) if np.any(ir_mask) else 0.0,
        },
    }

    return metrics


def format_marginal_report(metrics: Dict[str, Any]) -> str:
    """Formats human-readable text report."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("HIERARCHICAL MARGINAL EVALUATION REPORT (UFPR-VeSV)")
    report.append("=" * 80)
    report.append(f"Total Evaluated Samples: {metrics['total_samples']:,}")
    report.append(f"Visible Light (RGB): {metrics['rgb']['count']:,} | Infrared (IR): {metrics['ir']['count']:,}")
    report.append("-" * 80)
    report.append(f"{'Metric':<45} | {'Result (Marginal Decoding)':<25}")
    report.append("-" * 80)
    report.append(f"{'Type Accuracy (14 classes)':<45} | {metrics['acc_type']:>20.2f}%")
    report.append(f"{'Make Accuracy (26 classes)':<45} | {metrics['acc_make']:>20.2f}%")
    report.append(f"{'Model Accuracy (136 classes)':<45} | {metrics['acc_model']:>20.2f}%")
    report.append("-" * 80)
    report.append(f"{'[★] MARGINAL ACCURACY (Mean T+M+M)':<45} | {metrics['marginal_acc']:>20.2f}%")
    report.append(f"{'[★] Full Tuple Accuracy (Exact Match)':<45} | {metrics['acc_exact_match']:>20.2f}%")
    report.append("-" * 80)
    report.append(f"{'[!] Inconsistency Make <-> Model':<45} | {metrics['pct_invalid_make_model']:>20.2f}%")
    report.append(f"{'[!] Inconsistency Model <-> Type':<45} | {metrics['pct_invalid_model_type']:>20.2f}%")
    report.append(f"{'[!] TOTAL INVALID TUPLES RATE':<45} | {metrics['pct_invalid_total']:>20.2f}%")
    report.append("-" * 80)
    report.append(f"{'Tuple Accuracy RGB (Visible Light)':<45} | {metrics['rgb']['acc_exact']:>20.2f}%")
    report.append(f"{'Tuple Accuracy IR (Infrared)':<45} | {metrics['ir']['acc_exact']:>20.2f}%")
    report.append("=" * 80 + "\n")
    return "\n".join(report)


def print_marginal_report(metrics: Dict[str, Any]) -> None:
    text = format_marginal_report(metrics)
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical Marginal Evaluation & Invalid Tuple Assessment on UFPR-VeSV"
    )
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Root directory of UFPR-VeSV dataset")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to extracted embeddings .pt file")
    parser.add_argument("--backbone", type=str, default="", help="Backbone model architecture (auto-detected from embeddings file if omitted)")
    parser.add_argument("--split_fold", type=int, default=-1, help="Split fold (auto-detected from embeddings file if omitted)")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs to train linear probe")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate for linear probe")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for linear probe")
    parser.add_argument("--hidden_dim", type=int, default=0, help="Hidden dimension for MLP probe (0 = Linear probe)")
    parser.add_argument("--test_subset", type=str, default="test", choices=["test", "val"], help="Subset to evaluate")
    parser.add_argument("--include_hcdc", action="store_true", help="Include HCD-C constrained decoding in output")
    parser.add_argument("--exp_name", type=str, default="", help="Custom experiment name for logging in results/")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory where experiment logs are stored")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 1. Build Taxonomy
    data_dir = Path(args.data_dir)
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")
    print("=" * 80)
    print("UFPR-VeSV VEHICLE TAXONOMY CONSTRUCTED:")
    print(f" - Type Classes:  {taxonomy.num_types}")
    print(f" - Make Classes:  {taxonomy.num_makes}")
    print(f" - Model Classes: {taxonomy.num_models}")
    print(f" - Cartesian Product Space: {taxonomy.cartesian_size:,} combinations")
    print(f" - Ground Truth Valid Tuples: {taxonomy.num_valid_tuples} ({taxonomy.num_valid_tuples/taxonomy.cartesian_size*100:.2f}% of space)")
    print("=" * 80)

    # 2. Load extracted embeddings
    emb_path = Path(args.embeddings_file)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    data_pt = torch.load(emb_path, map_location="cpu")

    # Auto-resolve backbone and split_fold from metadata or path if not explicitly provided
    metadata = data_pt.get("metadata", {}) if isinstance(data_pt, dict) else {}
    if not args.backbone:
        if metadata.get("backbone"):
            args.backbone = str(metadata["backbone"])
        else:
            parent_name = emb_path.parent.name
            args.backbone = parent_name if parent_name not in ["extracted_embeddings", "."] else emb_path.stem

    if args.split_fold == -1:
        if "split_fold" in metadata and metadata["split_fold"] is not None:
            args.split_fold = int(metadata["split_fold"])
        else:
            import re
            match = re.search(r"fold_?(\d+)", str(emb_path))
            if match:
                args.split_fold = int(match.group(1))
            else:
                args.split_fold = 0

    train_data = data_pt["train"]
    test_data = data_pt[args.test_subset]

    train_embeddings = train_data["embeddings"]
    train_targets = {
        "type": train_data["targets_type"],
        "make": train_data["targets_make"],
        "model": train_data["targets_model"],
    }

    test_embeddings = test_data["embeddings"]
    test_targets = {
        "type": test_data["targets_type"],
        "make": test_data["targets_make"],
        "model": test_data["targets_model"],
    }
    test_is_ir = test_data["is_infrared"]

    # 3. Train Probe
    probe = train_linear_probe(
        train_embeddings=train_embeddings,
        train_targets=train_targets,
        val_embeddings=test_embeddings,
        val_targets=test_targets,
        taxonomy=taxonomy,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    # 4. Inference
    probe.eval()
    with torch.no_grad():
        feats_test = test_embeddings.to(device)
        logits_type, logits_make, logits_model = probe(feats_test)

        probs_type = F.softmax(logits_type, dim=-1)
        probs_make = F.softmax(logits_make, dim=-1)
        probs_model = F.softmax(logits_model, dim=-1)

        pred_type_std = torch.argmax(probs_type, dim=-1)
        pred_make_std = torch.argmax(probs_make, dim=-1)
        pred_model_std = torch.argmax(probs_model, dim=-1)

        marginal_metrics = compute_hierarchical_metrics(
            pred_type=pred_type_std,
            pred_make=pred_make_std,
            pred_model=pred_model_std,
            true_type=test_targets["type"].to(device),
            true_make=test_targets["make"].to(device),
            true_model=test_targets["model"].to(device),
            is_infrared=test_is_ir.to(device),
            taxonomy=taxonomy,
            prefix="Marginal Decoding",
        )

    # 5. Output and Automatic Results Logging
    report_text = format_marginal_report(marginal_metrics)
    print(report_text)

    exp_name = args.exp_name if args.exp_name else f"eval_{args.backbone}_fold{args.split_fold}"
    save_experiment_result(
        experiment_name=exp_name,
        config=args,
        metrics=marginal_metrics,
        report_text=report_text,
        output_root=args.output_dir,
    )


if __name__ == "__main__":
    main()
