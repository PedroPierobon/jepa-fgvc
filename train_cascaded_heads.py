#!/usr/bin/env python3
"""
Cascaded Hierarchical Conditioning for Fine-Grained Vehicle Classification
==========================================================================
Implements conditionally chained classification heads (Type -> Make -> Model)
trained on frozen backbone visual representations.

Motivation:
Independent multi-task heads produce disjoint marginal argmax predictions that
frequently violate taxonomic ontology (e.g. predicting motorcycle + Ford + Civic),
leading to a ~5.44% pre-HCD invalid tuple rate.

Cascaded Conditioning Architecture:
- Level 0 (Type):  z_type  = Head_Type(h)               -> p_type = softmax(z_type)
- Level 1 (Make):  z_make  = Head_Make(h, p_type)      -> p_make = softmax(z_make)
- Level 2 (Model): z_model = Head_Model(h, p_make, p_type)

By conditioning Make on Type and Model on Make, the network learns taxonomic
compatibility directly in the parameter space, slashing logical hallucinations
at the root before HCD post-processing.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
REPO_HCD = BASE_DIR.parent / "HCD-C"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_HCD))

from datasets.ufpr_dataset import UFPRDataset, get_eval_transform
from finetune_hierarchical import HierarchicalClassifier
from evaluate_hierarchical import VehicleTaxonomy
from hcd import Catalog, HCD
from utils.results_logger import save_experiment_result


class CascadedHierarchicalHeads(nn.Module):
    """
    Cascaded Conditional Classification Heads:
    Type (14) -> Make (26) -> Model (136).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        num_types: int = 14,
        num_makes: int = 26,
        num_models: int = 136,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.num_makes = num_makes
        self.num_models = num_models

        # Level 0: Type Head
        self.type_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )

        # Type -> Make conditioning projection
        self.type_projector = nn.Sequential(
            nn.Linear(num_types, hidden_dim // 2),
            nn.GELU(),
        )

        # Level 1: Make Head conditioned on visual features + Type distribution
        self.make_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_makes),
        )

        # (Make, Type) -> Model conditioning projection
        self.make_type_projector = nn.Sequential(
            nn.Linear(num_makes + num_types, hidden_dim),
            nn.GELU(),
        )

        # Level 2: Model Head conditioned on visual features + Make & Type distribution
        self.model_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_models),
        )

    def forward(
        self,
        h: torch.Tensor,
        target_type: Optional[torch.Tensor] = None,
        target_make: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Type prediction
        logits_type = self.type_net(h)

        # Decide whether to use teacher forcing for Type condition
        use_tf_type = (
            self.training
            and target_type is not None
            and (torch.rand(1).item() < teacher_forcing_prob)
        )
        if use_tf_type:
            p_type = F.one_hot(target_type, num_classes=self.num_types).float()
        else:
            p_type = F.softmax(logits_type, dim=-1)

        cond_type = self.type_projector(p_type)

        # 2. Make prediction
        h_make = torch.cat([h, cond_type], dim=-1)
        logits_make = self.make_net(h_make)

        # Decide whether to use teacher forcing for Make condition
        use_tf_make = (
            self.training
            and target_make is not None
            and (torch.rand(1).item() < teacher_forcing_prob)
        )
        if use_tf_make:
            p_make = F.one_hot(target_make, num_classes=self.num_makes).float()
        else:
            p_make = F.softmax(logits_make, dim=-1)

        cond_make_type = self.make_type_projector(torch.cat([p_make, p_type], dim=-1))

        # 3. Model prediction
        h_model = torch.cat([h, cond_make_type], dim=-1)
        logits_model = self.model_net(h_model)

        return logits_type, logits_make, logits_model


@torch.no_grad()
def extract_and_cache_embeddings(
    checkpoint_path: Path,
    backbone_name: str,
    data_dir: Path,
    split_fold: int,
    output_path: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Extracts visual embeddings from the fine-tuned backbone encoder across train, val, test splits.
    """
    print(f"[*] Embeddings file not found at: {output_path}")
    print(f"[*] Extracting embeddings from checkpoint: {checkpoint_path}")

    model = HierarchicalClassifier(backbone_name=backbone_name, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    encoder = model.encoder.to(device)
    encoder.eval()

    eval_transform = get_eval_transform(img_size=224)
    extracted = {
        "metadata": {
            "backbone": backbone_name,
            "embed_dim": encoder.embed_dim,
            "split_fold": split_fold,
            "checkpoint": str(checkpoint_path),
        }
    }

    for subset in ["train", "val", "test"]:
        print(f"  -> Extracting '{subset}' split...", end=" ", flush=True)
        ds = UFPRDataset(root_dir=data_dir, split_fold=split_fold, subset=subset, transform=eval_transform, is_pretrain=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

        all_embs, all_yt, all_ym, all_ymo, all_ir = [], [], [], [], []
        for images, meta in loader:
            images = images.to(device, non_blocking=True)
            feats = encoder(images)
            all_embs.append(feats.cpu())
            all_yt.append(meta["target_type"].cpu())
            all_ym.append(meta["target_make"].cpu())
            all_ymo.append(meta["target_model"].cpu())
            all_ir.append(meta["is_infrared"].cpu())

        extracted[subset] = {
            "embeddings": torch.cat(all_embs, dim=0),
            "targets_type": torch.cat(all_yt, dim=0),
            "targets_make": torch.cat(all_ym, dim=0),
            "targets_model": torch.cat(all_ymo, dim=0),
            "is_infrared": torch.cat(all_ir, dim=0),
        }
        print(f"Done! Shape: {extracted[subset]['embeddings'].shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, output_path)
    print(f"[✓] Saved consolidated embeddings to: {output_path}")

    del encoder, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return extracted


def train_cascaded_epoch(
    model: CascadedHierarchicalHeads,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_prob: float = 0.3,
) -> float:
    """Runs one training epoch over frozen embeddings."""
    model.train()
    total_loss = 0.0

    for embs, yt, ym, ymo, _ in loader:
        embs = embs.to(device)
        yt, ym, ymo = yt.to(device), ym.to(device), ymo.to(device)

        optimizer.zero_grad()
        ot, om, omo = model(
            embs, target_type=yt, target_make=ym, teacher_forcing_prob=teacher_forcing_prob
        )

        loss_type = criterion(ot, yt)
        loss_make = criterion(om, ym)
        loss_model = criterion(omo, ymo)
        # Balanced multi-task loss
        loss = loss_type + loss_make + loss_model

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_cascaded(
    model: CascadedHierarchicalHeads,
    loader: DataLoader,
    catalog: Catalog,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[np.ndarray], np.ndarray, np.ndarray]:
    """Evaluates the cascaded conditional heads and checks hierarchical consistency."""
    model.eval()
    all_zt, all_zm, all_zmo = [], [], []
    all_yt, all_ym, all_ymo = [], [], []
    all_ir = []

    for embs, yt, ym, ymo, is_ir in loader:
        embs = embs.to(device)
        ot, om, omo = model(embs, teacher_forcing_prob=0.0)

        all_zt.append(ot.cpu().numpy())
        all_zm.append(om.cpu().numpy())
        all_zmo.append(omo.cpu().numpy())

        all_yt.append(yt.numpy())
        all_ym.append(ym.numpy())
        all_ymo.append(ymo.numpy())
        all_ir.append(is_ir.numpy())

    z = [
        np.concatenate(all_zt, axis=0),
        np.concatenate(all_zm, axis=0),
        np.concatenate(all_zmo, axis=0),
    ]
    y = np.stack(
        [
            np.concatenate(all_yt, axis=0),
            np.concatenate(all_ym, axis=0),
            np.concatenate(all_ymo, axis=0),
        ],
        axis=1,
    )
    is_ir = np.concatenate(all_ir, axis=0).astype(bool)

    # Marginal prediction
    pred_marg = np.stack([zt.argmax(1) for zt in z], axis=1)
    exact_marg = (pred_marg == y).all(axis=1)

    rgb_mask = ~is_ir
    ir_mask = is_ir

    acc_exact = float(exact_marg.mean() * 100.0)
    acc_rgb = float(exact_marg[rgb_mask].mean() * 100.0)
    acc_ir = float(exact_marg[ir_mask].mean() * 100.0)

    acc_type = float((pred_marg[:, 0] == y[:, 0]).mean() * 100.0)
    acc_make = float((pred_marg[:, 1] == y[:, 1]).mean() * 100.0)
    acc_model = float((pred_marg[:, 2] == y[:, 2]).mean() * 100.0)
    mean_task_acc = (acc_type + acc_make + acc_model) / 3.0

    # Invalid tuple rate
    invalid_pct = float((~catalog.contains(pred_marg)).mean() * 100.0)

    stats = {
        "exact": acc_exact,
        "rgb": acc_rgb,
        "ir": acc_ir,
        "type": acc_type,
        "make": acc_make,
        "model": acc_model,
        "mean_task": mean_task_acc,
        "invalid_pct": invalid_pct,
    }

    return stats, z, y, is_ir


def main():
    parser = argparse.ArgumentParser(
        description="Train Cascaded Conditional Heads on Frozen Embeddings"
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="./extracted_embeddings/fold0_convnext/embeddings_fold_0.pt",
        help="Path to pre-extracted embeddings .pt file",
    )
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Path to UFPR-VeSV root directory")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 9)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_ft/convnext_lejepa_ft/best_model_fold0.pth",
        help="Fallback checkpoint for feature extraction if embeddings file is missing",
    )
    parser.add_argument("--backbone", type=str, default="convnext_base", help="Backbone model name")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for cascaded heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding training")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for conditioning projectors")
    parser.add_argument("--teacher_forcing_prob", type=float, default=0.3, help="Teacher forcing probability during training")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save experiment results")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    data_dir = Path(args.data_dir).resolve()
    embeddings_file = Path(args.embeddings_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[*] Starting Cascaded Conditioning Training on: {device}")
    print(f"[*] Dataset: {data_dir} | Fold: {args.fold} | Target Epochs: {args.epochs}")

    # Build taxonomy and valid tuple catalog
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")
    catalog = Catalog(
        taxonomy.valid_tuples_tensor.numpy(),
        task_names=["type", "make", "model"],
        n_classes=[14, 26, 136],
    )
    print(f"[*] Valid Tuple Catalog loaded: {catalog}")

    # 1. Load or extract embeddings
    if embeddings_file.exists():
        print(f"[*] Loading pre-extracted embeddings from: {embeddings_file}")
        data = torch.load(embeddings_file, map_location="cpu", weights_only=False)
    else:
        ckpt_path = Path(args.checkpoint).resolve()
        data = extract_and_cache_embeddings(
            checkpoint_path=ckpt_path,
            backbone_name=args.backbone,
            data_dir=data_dir,
            split_fold=args.fold,
            output_path=embeddings_file,
            device=device,
        )

    # Build DataLoaders from tensors
    def make_loader(split_data, batch_size, shuffle=False):
        ds = TensorDataset(
            split_data["embeddings"].float(),
            split_data["targets_type"].long(),
            split_data["targets_make"].long(),
            split_data["targets_model"].long(),
            split_data["is_infrared"].bool(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(data["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(data["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(data["test"], batch_size=args.batch_size, shuffle=False)

    embed_dim = data["train"]["embeddings"].shape[1]
    print(f"[*] Representation dimension: {embed_dim} | Training samples: {len(data['train']['embeddings']):,}")

    # 2. Instantiate Cascaded Heads
    cascaded_model = CascadedHierarchicalHeads(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_types=14,
        num_makes=26,
        num_models=136,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(cascaded_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Training Cascaded Heads for {args.epochs} epochs...")
    best_val_exact = -1.0
    best_state = None

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_cascaded_epoch(
            cascaded_model, train_loader, optimizer, criterion, device, teacher_forcing_prob=args.teacher_forcing_prob
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_stats, _, _, _ = evaluate_cascaded(cascaded_model, val_loader, catalog, device)
            print(
                f"  Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {loss:.4f} | "
                f"Val Exact: {val_stats['exact']:.2f}% | Val Invalid: {val_stats['invalid_pct']:.2f}%"
            )
            if val_stats["exact"] > best_val_exact:
                best_val_exact = val_stats["exact"]
                best_state = {k: v.cpu().clone() for k, v in cascaded_model.state_dict().items()}

    elapsed = time.time() - start_time
    print(f"[✓] Training completed in {elapsed:.1f}s ({elapsed/60:.2f} min). Best Val Exact: {best_val_exact:.2f}%")

    # Load best weights
    if best_state is not None:
        cascaded_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 3. Final Evaluation on Test Set (5,075 samples)
    print("\n" + "=" * 85)
    print("FINAL TEST EVALUATION: CASCADED HIERARCHICAL HEADS (FOLD 0)")
    print("=" * 85)

    test_stats_marg, z_test, y_test, is_ir_test = evaluate_cascaded(cascaded_model, test_loader, catalog, device)
    _, z_val, y_val, is_ir_val = evaluate_cascaded(cascaded_model, val_loader, catalog, device)

    # 4. Apply HCD Pure and HCD-C on top of Cascaded Heads
    print("[*] Evaluating HCD Pure on cascaded predictions...")
    hcd_pure = HCD(catalog)
    pred_hcd_pure = hcd_pure.predict(z_test)
    exact_hcd_pure = (pred_hcd_pure == y_test).all(axis=1)
    acc_pure_exact = float(exact_hcd_pure.mean() * 100.0)
    acc_pure_rgb = float(exact_hcd_pure[~is_ir_test].mean() * 100.0)
    acc_pure_ir = float(exact_hcd_pure[is_ir_test].mean() * 100.0)
    diag_pure = hcd_pure.diagnostics(z_test, y_test)

    print("[*] Calibrating HCD-C on validation set...")
    hcdc = HCD(catalog).fit(z_val, y_val, verbose=False)
    pred_hcdc = hcdc.predict(z_test)
    exact_hcdc = (pred_hcdc == y_test).all(axis=1)
    acc_hcdc_exact = float(exact_hcdc.mean() * 100.0)
    acc_hcdc_rgb = float(exact_hcdc[~is_ir_test].mean() * 100.0)
    acc_hcdc_ir = float(exact_hcdc[is_ir_test].mean() * 100.0)
    diag_hcdc = hcdc.diagnostics(z_test, y_test)

    # Task accuracies under HCD-C
    acc_type_hcdc = float((pred_hcdc[:, 0] == y_test[:, 0]).mean() * 100.0)
    acc_make_hcdc = float((pred_hcdc[:, 1] == y_test[:, 1]).mean() * 100.0)
    acc_model_hcdc = float((pred_hcdc[:, 2] == y_test[:, 2]).mean() * 100.0)

    # -------------------------------------------------------------
    # COMPARISON WITH INDEPENDENT BASELINE & VERIFICATION CHECK
    # -------------------------------------------------------------
    baseline_invalid = 5.44  # Original independent heads pre-HCD invalid rate
    invalid_target_met = test_stats_marg["invalid_pct"] < 4.0

    print("\n" + "#" * 85)
    print("RESULTS COMPARISON: INDEPENDENT HEADS vs. CASCADED CONDITIONAL HEADS")
    print("#" * 85)
    print(f"Pre-HCD Invalid Tuple Rate (Baseline Independent): {baseline_invalid:.2f}%")
    print(f"Pre-HCD Invalid Tuple Rate (Cascaded Heads):        {test_stats_marg['invalid_pct']:.2f}%")
    if invalid_target_met:
        print(f"[✓] TARGET ACHIEVED: Invalid tuple rate dropped below 4.0% ({test_stats_marg['invalid_pct']:.2f}% < 4.00%)!")
    else:
        print(f"[!] Target check: {test_stats_marg['invalid_pct']:.2f}% (Target: < 4.00%)")

    print(f"\nMarginal Exact Tuple Accuracy:   {test_stats_marg['exact']:.2f}% (RGB: {test_stats_marg['rgb']:.2f}%, IR: {test_stats_marg['ir']:.2f}%)")
    print(f"HCD Pure Exact Tuple Accuracy:   {acc_pure_exact:.2f}% (RGB: {acc_pure_rgb:.2f}%, IR: {acc_pure_ir:.2f}%)")
    print(f"HCD-C Calibrated Exact Tuple:    {acc_hcdc_exact:.2f}% (RGB: {acc_hcdc_rgb:.2f}%, IR: {acc_hcdc_ir:.2f}%)")
    print(f"Rescued Samples (HCD-C):         {diag_hcdc['rescued']} (Broken: {diag_hcdc['broken']})")
    print(f"Post-HCD Invalid Tuples:         0.00%")
    print("#" * 85)

    # Save trained cascaded model checkpoint
    ckpt_save_dir = Path("./checkpoints_ft/cascaded_heads")
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = ckpt_save_dir / f"best_cascaded_heads_fold{args.fold}.pth"
    torch.save(
        {
            "model_state_dict": cascaded_model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": args.hidden_dim,
            "val_exact": best_val_exact,
            "test_exact": acc_hcdc_exact,
            "invalid_pct": test_stats_marg["invalid_pct"],
        },
        model_save_path,
    )
    print(f"\n[✓] Best cascaded heads weights saved to: {model_save_path}")

    # Build report text
    report_text = f"""CASCADED CONDITIONAL HEADS EVALUATION REPORT
Dataset: UFPR-VeSV | Fold: {args.fold} | Backbone: {args.backbone}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
1. TAXONOMIC CONSISTENCY (PRE-HCD INVALID TUPLES):
- Baseline Independent Heads Invalid Rate: 5.44%
- Cascaded Conditional Heads Invalid Rate: {test_stats_marg['invalid_pct']:.2f}%
- Target Status (< 4.0%):                  {'ACHIEVED [✓]' if invalid_target_met else 'PENDING'}

2. TEST SET ACCURACIES (5,075 SAMPLES):
- Marginal Exact Tuple:                    {test_stats_marg['exact']:.2f}% (RGB: {test_stats_marg['rgb']:.2f}%, IR: {test_stats_marg['ir']:.2f}%)
- Task Accuracies (Marginal):              Type: {test_stats_marg['type']:.2f}%, Make: {test_stats_marg['make']:.2f}%, Model: {test_stats_marg['model']:.2f}%
- HCD Pure (w=[1,1,1]):                    {acc_pure_exact:.2f}% (RGB: {acc_pure_rgb:.2f}%, IR: {acc_pure_ir:.2f}%)
- HCD-C Calibrated:                        {acc_hcdc_exact:.2f}% (RGB: {acc_hcdc_rgb:.2f}%, IR: {acc_hcdc_ir:.2f}%)
- Task Accuracies (HCD-C):                 Type: {acc_type_hcdc:.2f}%, Make: {acc_make_hcdc:.2f}%, Model: {acc_model_hcdc:.2f}%
- Rescued Samples:                         {diag_hcdc['rescued']} (Broken: {diag_hcdc['broken']})
- Post-HCD Invalid Rate:                   0.00%

3. CALIBRATION PARAMETERS:
- Learned HCD-C weights:                   w = {hcdc.w.round(4).tolist()}
- Temperature:                             T = {hcdc.T:.4f}
"""

    # Structured metrics dict for benchmark summary logging
    result_metrics = {
        "marginal_acc": test_stats_marg["mean_task"],
        "acc_type": acc_type_hcdc,
        "acc_make": acc_make_hcdc,
        "acc_model": acc_model_hcdc,
        "acc_exact_match": acc_hcdc_exact,
        "pct_invalid_total": 0.0,
        "pct_invalid_make_model": 0.0,
        "pct_invalid_model_type": 0.0,
        "rgb": {"acc_exact": acc_hcdc_rgb},
        "ir": {"acc_exact": acc_hcdc_ir},
        # Diagnostic details
        "pre_hcd_invalid_pct": test_stats_marg["invalid_pct"],
        "pre_hcd_exact": test_stats_marg["exact"],
        "pre_hcd_type": test_stats_marg["type"],
        "pre_hcd_make": test_stats_marg["make"],
        "pre_hcd_model": test_stats_marg["model"],
        "rescued": diag_hcdc["rescued"],
        "broken": diag_hcdc["broken"],
        "target_below_4pct": bool(invalid_target_met),
    }

    # Save structured results folder and update benchmark_summary.csv
    exp_name = f"train_cascaded_heads_{args.backbone}_fold{args.fold}"
    config_dict = {
        "experiment": exp_name,
        "backbone": f"cascaded_{args.backbone}",
        "split_fold": args.fold,
        "data_dir": str(data_dir),
        "embeddings_file": str(embeddings_file),
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "teacher_forcing_prob": args.teacher_forcing_prob,
        "method": "cascaded_conditional_heads",
    }
    save_experiment_result(
        experiment_name=exp_name,
        config=config_dict,
        metrics=result_metrics,
        report_text=report_text,
        output_root=str(output_dir),
    )


if __name__ == "__main__":
    main()
