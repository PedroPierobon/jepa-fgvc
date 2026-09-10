#!/usr/bin/env python3
"""
Bimodal Temperature and Task Calibration for HCD-C (UFPR-VeSV)
==============================================================
Evaluates Hierarchical Consistent Decoding with Calibrated parameters (HCD-C)
under bimodal environmental conditions:
1. Visible Light / Daylight (RGB)
2. Active Infrared / Nighttime (IR)

Surveillance cameras experience significant sensor noise, contrast reduction,
and loss of chromatic cues at night in active IR mode. This script compares:
- Baseline Marginal Decoding (Independent Argmax per task)
- HCD Pure (w=[1, 1, 1], T=1.0)
- HCD-C Unimodal (Single global w and T calibrated across all validation samples)
- HCD-C Bimodal Temperature (Shared task weights, separate T_RGB and T_IR)
- HCD-C Bimodal Full (Separate task weights w_RGB, w_IR and temperatures T_RGB, T_IR)
- HCD-C Task-Temperature Bimodal (Independent per-task temperature scaling for RGB and IR)

Outputs comprehensive metrics:
- Exact joint tuple accuracy (Overall, Daylight RGB, Infrared IR)
- Per-task marginal accuracy (Type, Make, Model)
- Invalid tuple rate (Hierarchical consistency violations)
- Rescued samples vs. Broken samples
- Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL)
- Automatic saving of config.json, metrics.json, report.txt, and update of benchmark_summary.csv.
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
from torch.utils.data import DataLoader
from scipy.optimize import minimize

# Add project root and HCD-C repository to python path
BASE_DIR = Path(__file__).resolve().parent
REPO_HCD = BASE_DIR.parent / "HCD-C"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_HCD))

from datasets.ufpr_dataset import UFPRDataset, get_eval_transform
from finetune_hierarchical import HierarchicalClassifier
from evaluate_hierarchical import VehicleTaxonomy
from hcd import Catalog, HCD, logsumexp
from utils.results_logger import save_experiment_result


def compute_ece(probs: np.ndarray, labels_idx: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) for top-1 predictions.
    
    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        labels_idx: Array of shape (N,) containing ground truth class indices.
        n_bins: Number of confidence bins (default: 15).
        
    Returns:
        Expected Calibration Error in percentage (0 to 100).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels_idx)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece * 100.0)


def compute_nll(probs: np.ndarray, labels_idx: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes average Negative Log-Likelihood (NLL) of the true class.
    
    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        labels_idx: Array of shape (N,) containing ground truth class indices.
        eps: Small epsilon to prevent log(0).
        
    Returns:
        Average negative log-likelihood.
    """
    valid_mask = (labels_idx >= 0) & (labels_idx < probs.shape[1])
    if not np.any(valid_mask):
        return float("nan")
    
    p = probs[valid_mask, labels_idx[valid_mask]]
    p_clipped = np.clip(p, eps, 1.0)
    return float(-np.mean(np.log(p_clipped)))


@torch.no_grad()
def extract_or_load_logits(
    model: Optional[torch.nn.Module],
    dataset: UFPRDataset,
    cache_path: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device = torch.device("cuda"),
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Extracts classifier logits and targets from dataset, with disk caching support.
    
    Returns:
        z: List of 3 numpy arrays [zt (N, 14), zm (N, 26), zmo (N, 136)]
        y: Array of shape (N, 3) containing integer targets [type, make, model]
        is_ir: Boolean array of shape (N,) indicating whether image is infrared
    """
    if cache_path and cache_path.exists():
        print(f"[*] Loading cached logits from: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["z"], data["y"], data["is_ir"]

    if model is None:
        raise ValueError("Model must be provided when logits cache does not exist.")

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_zt, all_zm, all_zmo = [], [], []
    all_yt, all_ym, all_ymo = [], [], []
    all_ir = []

    for images, meta in loader:
        images = images.to(device, non_blocking=True)
        ot, om, omo = model(images)

        all_zt.append(ot.cpu().numpy())
        all_zm.append(om.cpu().numpy())
        all_zmo.append(omo.cpu().numpy())

        all_yt.append(meta["target_type"].numpy())
        all_ym.append(meta["target_make"].numpy())
        all_ymo.append(meta["target_model"].numpy())
        all_ir.append(meta["is_infrared"].numpy())

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

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z": z, "y": y, "is_ir": is_ir}, cache_path)
        print(f"[✓] Logits saved to cache: {cache_path}")

    return z, y, is_ir


def fit_temperature_only(
    catalog: Catalog,
    z_val: List[np.ndarray],
    y_val: np.ndarray,
    w: np.ndarray,
    maxiter: int = 1000,
) -> float:
    """
    Optimizes a single temperature T minimizing joint NLL on validation set with fixed weights w.
    """
    y_idx = catalog.index_of(y_val)
    keep = y_idx >= 0
    if not keep.any():
        return 1.0
    
    z_val_sub = [zt[keep] for zt in z_val]
    y_idx_sub = y_idx[keep]
    n = np.arange(len(y_idx_sub))
    TUP = catalog.tuples

    # Unscaled joint score
    base_score = w[0] * z_val_sub[0][:, TUP[:, 0]]
    for t in range(1, catalog.n_tasks):
        base_score = base_score + w[t] * z_val_sub[t][:, TUP[:, t]]

    def obj(p):
        T = float(np.exp(p[0]))
        s = base_score / T
        logp = s - logsumexp(s, axis=1, keepdims=True)
        return float(-logp[n, y_idx_sub].mean())

    p0 = np.array([0.0])  # log(1.0) = 0
    res = minimize(obj, p0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=maxiter))
    best_T = float(np.exp(res.x[0]))
    return best_T


def fit_task_temperatures(
    z_val_task: np.ndarray,
    y_val_task: np.ndarray,
    maxiter: int = 1000,
) -> float:
    """
    Optimizes scalar temperature for a single task using cross-entropy on validation subset.
    """
    valid = (y_val_task >= 0) & (y_val_task < z_val_task.shape[1])
    if not valid.any():
        return 1.0

    z_sub = z_val_task[valid]
    y_sub = y_val_task[valid]
    n = np.arange(len(y_sub))

    def obj(p):
        T = float(np.exp(p[0]))
        s = z_sub / T
        logp = s - logsumexp(s, axis=1, keepdims=True)
        return float(-logp[n, y_sub].mean())

    p0 = np.array([0.0])
    res = minimize(obj, p0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=maxiter))
    return float(np.exp(res.x[0]))


class BimodalHCDC:
    """
    Bimodal Hierarchical Consistent Decoder.
    Maintains distinct task weights and temperatures for Daylight (RGB) and Nighttime (IR) conditions.
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.hcd_rgb = HCD(catalog)
        self.hcd_ir = HCD(catalog)
        self.fitted = False

    def fit(
        self,
        z_val: List[np.ndarray],
        y_val: np.ndarray,
        is_ir_val: np.ndarray,
        verbose: bool = True,
    ) -> "BimodalHCDC":
        rgb_mask = ~is_ir_val
        ir_mask = is_ir_val

        z_val_rgb = [zt[rgb_mask] for zt in z_val]
        y_val_rgb = y_val[rgb_mask]

        z_val_ir = [zt[ir_mask] for zt in z_val]
        y_val_ir = y_val[ir_mask]

        if verbose:
            print(f"[*] Calibrating RGB split ({rgb_mask.sum()} samples)...")
        self.hcd_rgb.fit(z_val_rgb, y_val_rgb, verbose=verbose)

        if verbose:
            print(f"[*] Calibrating IR split ({ir_mask.sum()} samples)...")
        self.hcd_ir.fit(z_val_ir, y_val_ir, verbose=verbose)

        self.fitted = True
        return self

    def predict(self, z: List[np.ndarray], is_ir: np.ndarray) -> np.ndarray:
        rgb_mask = ~is_ir
        ir_mask = is_ir
        n_samples = z[0].shape[0]
        preds = np.zeros((n_samples, self.catalog.n_tasks), dtype=np.int64)

        if np.any(rgb_mask):
            z_rgb = [zt[rgb_mask] for zt in z]
            preds[rgb_mask] = self.hcd_rgb.predict(z_rgb)

        if np.any(ir_mask):
            z_ir = [zt[ir_mask] for zt in z]
            preds[ir_mask] = self.hcd_ir.predict(z_ir)

        return preds

    def predict_proba(self, z: List[np.ndarray], is_ir: np.ndarray) -> np.ndarray:
        rgb_mask = ~is_ir
        ir_mask = is_ir
        n_samples = z[0].shape[0]
        probs = np.zeros((n_samples, len(self.catalog)), dtype=float)

        if np.any(rgb_mask):
            z_rgb = [zt[rgb_mask] for zt in z]
            probs[rgb_mask] = self.hcd_rgb.predict_proba(z_rgb)

        if np.any(ir_mask):
            z_ir = [zt[ir_mask] for zt in z]
            probs[ir_mask] = self.hcd_ir.predict_proba(z_ir)

        return probs


def evaluate_backbone_bimodal(
    name: str,
    backbone: str,
    ckpt_path: Path,
    data_dir: Path,
    fold: int,
    catalog: Catalog,
    device: torch.device,
    cache_dir: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Evaluates a single fine-tuned backbone under all decoding regimes with Daylight/IR decomposition.
    """
    print(f"\n{'=' * 90}")
    print(f"EVALUATING MODEL: {name} ({backbone}) - Fold {fold}")
    print(f"{'=' * 90}")
    print(f"[*] Checkpoint path: {ckpt_path}")

    eval_transform = get_eval_transform(img_size=224)
    val_dataset = UFPRDataset(root_dir=data_dir, split_fold=fold, subset="val", transform=eval_transform, is_pretrain=False)
    test_dataset = UFPRDataset(root_dir=data_dir, split_fold=fold, subset="test", transform=eval_transform, is_pretrain=False)

    val_cache = (cache_dir / f"val_logits_{backbone}_fold{fold}.pt") if cache_dir else None
    test_cache = (cache_dir / f"test_logits_{backbone}_fold{fold}.pt") if cache_dir else None

    # Load model only if logits not already cached
    need_model = not (val_cache and val_cache.exists() and test_cache and test_cache.exists())
    model = None
    if need_model:
        print("[*] Instantiating model and loading weights...")
        model = HierarchicalClassifier(backbone_name=backbone, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

    print("[*] Extracting/retrieving validation logits...")
    z_val, y_val, is_ir_val = extract_or_load_logits(
        model, val_dataset, cache_path=val_cache, batch_size=batch_size, num_workers=num_workers, device=device
    )

    print("[*] Extracting/retrieving test logits (5,075 samples)...")
    z_test, y_test, is_ir_test = extract_or_load_logits(
        model, test_dataset, cache_path=test_cache, batch_size=batch_size, num_workers=num_workers, device=device
    )

    rgb_test_mask = ~is_ir_test
    ir_test_mask = is_ir_test
    rgb_val_mask = ~is_ir_val
    ir_val_mask = is_ir_val
    n_test = len(y_test)
    y_test_idx = catalog.index_of(y_test)

    # -------------------------------------------------------------
    # 1. Marginal Decoding Baseline (Independent argmax per task)
    # -------------------------------------------------------------
    pred_marg = np.stack([zt.argmax(1) for zt in z_test], axis=1)
    exact_marg = (pred_marg == y_test).all(axis=1)
    acc_exact_marg = float(exact_marg.mean() * 100.0)
    acc_rgb_marg = float(exact_marg[rgb_test_mask].mean() * 100.0)
    acc_ir_marg = float(exact_marg[ir_test_mask].mean() * 100.0)

    acc_type_marg = float((pred_marg[:, 0] == y_test[:, 0]).mean() * 100.0)
    acc_make_marg = float((pred_marg[:, 1] == y_test[:, 1]).mean() * 100.0)
    acc_model_marg = float((pred_marg[:, 2] == y_test[:, 2]).mean() * 100.0)
    marginal_mean_acc = (acc_type_marg + acc_make_marg + acc_model_marg) / 3.0
    invalid_pct_marg = float((~catalog.contains(pred_marg)).mean() * 100.0)

    # -------------------------------------------------------------
    # 2. HCD Pure (w = [1, 1, 1], T = 1.0)
    # -------------------------------------------------------------
    hcd_pure = HCD(catalog)
    pred_hcd_pure = hcd_pure.predict(z_test)
    exact_hcd_pure = (pred_hcd_pure == y_test).all(axis=1)
    acc_exact_pure = float(exact_hcd_pure.mean() * 100.0)
    acc_rgb_pure = float(exact_hcd_pure[rgb_test_mask].mean() * 100.0)
    acc_ir_pure = float(exact_hcd_pure[ir_test_mask].mean() * 100.0)
    diag_pure = hcd_pure.diagnostics(z_test, y_test)

    # -------------------------------------------------------------
    # 3. HCD-C Unimodal (Standard global calibration)
    # -------------------------------------------------------------
    print("\n[*] [Regime 3] Fitting standard Unimodal HCD-C on full validation set...")
    hcdc_uni = HCD(catalog).fit(z_val, y_val, verbose=False)
    pred_uni = hcdc_uni.predict(z_test)
    probs_uni = hcdc_uni.predict_proba(z_test)
    exact_uni = (pred_uni == y_test).all(axis=1)

    acc_exact_uni = float(exact_uni.mean() * 100.0)
    acc_rgb_uni = float(exact_uni[rgb_test_mask].mean() * 100.0)
    acc_ir_uni = float(exact_uni[ir_test_mask].mean() * 100.0)
    diag_uni = hcdc_uni.diagnostics(z_test, y_test)
    ece_uni_all = compute_ece(probs_uni, y_test_idx)
    ece_uni_rgb = compute_ece(probs_uni[rgb_test_mask], y_test_idx[rgb_test_mask])
    ece_uni_ir = compute_ece(probs_uni[ir_test_mask], y_test_idx[ir_test_mask])
    nll_uni_all = compute_nll(probs_uni, y_test_idx)

    # -------------------------------------------------------------
    # 4. HCD-C Bimodal Temperatures (Shared weights w_uni, separate T_RGB & T_IR)
    # -------------------------------------------------------------
    print("[*] [Regime 4] Calibrating bimodal temperatures (T_RGB vs T_IR) with shared weights...")
    t_rgb_temp = fit_temperature_only(catalog, [zt[rgb_val_mask] for zt in z_val], y_val[rgb_val_mask], w=hcdc_uni.w)
    t_ir_temp = fit_temperature_only(catalog, [zt[ir_val_mask] for zt in z_val], y_val[ir_val_mask], w=hcdc_uni.w)

    probs_bimodal_temp = np.zeros_like(probs_uni)
    if np.any(rgb_test_mask):
        probs_bimodal_temp[rgb_test_mask] = np.exp(hcdc_uni.log_proba([zt[rgb_test_mask] for zt in z_test], w=hcdc_uni.w, T=t_rgb_temp))
    if np.any(ir_test_mask):
        probs_bimodal_temp[ir_test_mask] = np.exp(hcdc_uni.log_proba([zt[ir_test_mask] for zt in z_test], w=hcdc_uni.w, T=t_ir_temp))

    ece_bim_temp_all = compute_ece(probs_bimodal_temp, y_test_idx)
    ece_bim_temp_rgb = compute_ece(probs_bimodal_temp[rgb_test_mask], y_test_idx[rgb_test_mask])
    ece_bim_temp_ir = compute_ece(probs_bimodal_temp[ir_test_mask], y_test_idx[ir_test_mask])
    nll_bim_temp_all = compute_nll(probs_bimodal_temp, y_test_idx)

    # -------------------------------------------------------------
    # 5. HCD-C Full Bimodal Calibration (w_RGB, T_RGB and w_IR, T_IR)
    # -------------------------------------------------------------
    print("[*] [Regime 5] Fitting Full Bimodal HCD-C (Independent w & T per modality)...")
    bim_hcdc = BimodalHCDC(catalog).fit(z_val, y_val, is_ir_val, verbose=False)
    pred_bim_full = bim_hcdc.predict(z_test, is_ir_test)
    probs_bim_full = bim_hcdc.predict_proba(z_test, is_ir_test)
    exact_bim_full = (pred_bim_full == y_test).all(axis=1)

    acc_exact_bim_full = float(exact_bim_full.mean() * 100.0)
    acc_rgb_bim_full = float(exact_bim_full[rgb_test_mask].mean() * 100.0)
    acc_ir_bim_full = float(exact_bim_full[ir_test_mask].mean() * 100.0)

    rescued_bim_full = int((~exact_marg & exact_bim_full).sum())
    broken_bim_full = int((exact_marg & ~exact_bim_full).sum())
    rescued_ir_full = int((~exact_marg[ir_test_mask] & exact_bim_full[ir_test_mask]).sum())
    rescued_rgb_full = int((~exact_marg[rgb_test_mask] & exact_bim_full[rgb_test_mask]).sum())

    ece_bim_full_all = compute_ece(probs_bim_full, y_test_idx)
    ece_bim_full_rgb = compute_ece(probs_bim_full[rgb_test_mask], y_test_idx[rgb_test_mask])
    ece_bim_full_ir = compute_ece(probs_bim_full[ir_test_mask], y_test_idx[ir_test_mask])
    nll_bim_full_all = compute_nll(probs_bim_full, y_test_idx)

    # Per-task accuracies under Full Bimodal HCD-C
    acc_type_bim = float((pred_bim_full[:, 0] == y_test[:, 0]).mean() * 100.0)
    acc_make_bim = float((pred_bim_full[:, 1] == y_test[:, 1]).mean() * 100.0)
    acc_model_bim = float((pred_bim_full[:, 2] == y_test[:, 2]).mean() * 100.0)

    # -------------------------------------------------------------
    # 6. Task-Specific Temperature Scaling per Modality
    # -------------------------------------------------------------
    print("[*] [Regime 6] Optimizing per-task temperature scaling for RGB vs IR...")
    task_temp_rgb = [
        fit_task_temperatures(z_val[t][rgb_val_mask], y_val[rgb_val_mask, t]) for t in range(3)
    ]
    task_temp_ir = [
        fit_task_temperatures(z_val[t][ir_val_mask], y_val[ir_val_mask, t]) for t in range(3)
    ]

    # Scale test logits
    z_scaled_test = [np.copy(zt) for zt in z_test]
    for t in range(3):
        z_scaled_test[t][rgb_test_mask] /= task_temp_rgb[t]
        z_scaled_test[t][ir_test_mask] /= task_temp_ir[t]

    hcd_task_scaled = HCD(catalog)
    pred_task_scaled = hcd_task_scaled.predict(z_scaled_test)
    exact_task_scaled = (pred_task_scaled == y_test).all(axis=1)
    acc_exact_task_scaled = float(exact_task_scaled.mean() * 100.0)
    acc_rgb_task_scaled = float(exact_task_scaled[rgb_test_mask].mean() * 100.0)
    acc_ir_task_scaled = float(exact_task_scaled[ir_test_mask].mean() * 100.0)

    # Summary table
    print("\n" + "=" * 92)
    print(f"DETAILED DECODING & CALIBRATION BREAKDOWN: {name} (Fold {fold})")
    print("=" * 92)
    print(f"{'Method / Calibration Regime':<32} | {'Joint Exact':<11} | {'RGB Exact':<10} | {'IR Exact':<10} | {'ECE (%)':<8} | {'Rescued'}")
    print("-" * 92)
    print(f"{'Marginal (Unconstrained Argmax)':<32} | {acc_exact_marg:>9.2f}% | {acc_rgb_marg:>8.2f}% | {acc_ir_marg:>8.2f}% | {'N/A':>8} | {'-':>7}")
    print(f"{'HCD Pure (w=[1,1,1], T=1.0)':<32} | {acc_exact_pure:>9.2f}% | {acc_rgb_pure:>8.2f}% | {acc_ir_pure:>8.2f}% | {'N/A':>8} | {diag_pure['rescued']:>7}")
    print(f"{'HCD-C Unimodal (Global Fit)':<32} | {acc_exact_uni:>9.2f}% | {acc_rgb_uni:>8.2f}% | {acc_ir_uni:>8.2f}% | {ece_uni_all:>7.2f}% | {diag_uni['rescued']:>7}")
    print(f"{'HCD-C Bimodal (T_RGB vs T_IR)':<32} | {acc_exact_uni:>9.2f}% | {acc_rgb_uni:>8.2f}% | {acc_ir_uni:>8.2f}% | {ece_bim_temp_all:>7.2f}% | {diag_uni['rescued']:>7}")
    print(f"{'HCD-C Bimodal Full (w & T)':<32} | {acc_exact_bim_full:>9.2f}% | {acc_rgb_bim_full:>8.2f}% | {acc_ir_bim_full:>8.2f}% | {ece_bim_full_all:>7.2f}% | {rescued_bim_full:>7}")
    print(f"{'HCD-C Task-Temperature Scaling':<32} | {acc_exact_task_scaled:>9.2f}% | {acc_rgb_task_scaled:>8.2f}% | {acc_ir_task_scaled:>8.2f}% | {'N/A':>8} | {(~exact_marg & exact_task_scaled).sum():>7}")
    print("=" * 92)

    print("\n--- PARAMETERS LEARNED IN CALIBRATION ---")
    print(f"Unimodal HCD-C:    w = {hcdc_uni.w.round(4).tolist()}, T = {hcdc_uni.T:.4f}")
    print(f"Bimodal Temp-Only: T_RGB = {t_rgb_temp:.4f}, T_IR = {t_ir_temp:.4f}")
    print(f"Bimodal Full:      w_RGB = {bim_hcdc.hcd_rgb.w.round(4).tolist()}, T_RGB = {bim_hcdc.hcd_rgb.T:.4f}")
    print(f"                   w_IR  = {bim_hcdc.hcd_ir.w.round(4).tolist()}, T_IR  = {bim_hcdc.hcd_ir.T:.4f}")
    print(f"Task-Temperatures: RGB (Type, Make, Model) = {[round(t, 4) for t in task_temp_rgb]}")
    print(f"                   IR  (Type, Make, Model) = {[round(t, 4) for t in task_temp_ir]}")

    print("\n--- INFRARED (IR) RESCUE & BENCHMARK COMPARISON ---")
    print(f"Baseline IR Accuracy (Previous literature / folds mean): 70.69%")
    print(f"Marginal IR Accuracy (Fold {fold}):                       {acc_ir_marg:.2f}%")
    print(f"HCD Pure IR Accuracy:                                    {acc_ir_pure:.2f}% (+{acc_ir_pure - acc_ir_marg:.2f} pp)")
    print(f"HCD-C Unimodal IR Accuracy:                              {acc_ir_uni:.2f}% (+{acc_ir_uni - acc_ir_marg:.2f} pp)")
    print(f"HCD-C Full Bimodal IR Accuracy:                          {acc_ir_bim_full:.2f}% (+{acc_ir_bim_full - acc_ir_marg:.2f} pp)")
    print(f"IR Samples Rescued by Bimodal HCD-C:                     {rescued_ir_full} / {ir_test_mask.sum()} IR images")
    print(f"RGB Samples Rescued by Bimodal HCD-C:                    {rescued_rgb_full} / {rgb_test_mask.sum()} RGB images")
    print(f"Total Samples Rescued:                                   {rescued_bim_full} (Broken: {broken_bim_full})")
    print(f"Net Accuracy Gain over Marginal:                         +{acc_exact_bim_full - acc_exact_marg:.2f} pp")

    # Format report string
    report_text = f"""BIMODAL TEMPERATURE AND TASK CALIBRATION REPORT (HCD-C)
Dataset: UFPR-VeSV | Fold: {fold} | Model: {name} ({backbone})
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
1. ACCURACY & CONSISTENCY SUMMARY:
- Marginal Exact Tuple Accuracy:   {acc_exact_marg:.2f}% (RGB: {acc_rgb_marg:.2f}%, IR: {acc_ir_marg:.2f}%)
- Marginal Task Accuracies:        Type: {acc_type_marg:.2f}%, Make: {acc_make_marg:.2f}%, Model: {acc_model_marg:.2f}%
- Pre-HCD Invalid Tuples:          {invalid_pct_marg:.2f}% (Violations eliminated post-HCD -> 0.00%)
- HCD Pure (w=[1,1,1]):            {acc_exact_pure:.2f}% (RGB: {acc_rgb_pure:.2f}%, IR: {acc_ir_pure:.2f}%)
- HCD-C Unimodal:                  {acc_exact_uni:.2f}% (RGB: {acc_rgb_uni:.2f}%, IR: {acc_ir_uni:.2f}%)
- HCD-C Bimodal Full:              {acc_exact_bim_full:.2f}% (RGB: {acc_rgb_bim_full:.2f}%, IR: {acc_ir_bim_full:.2f}%)
- HCD-C Task-Temperature Scaling:  {acc_exact_task_scaled:.2f}% (RGB: {acc_rgb_task_scaled:.2f}%, IR: {acc_ir_task_scaled:.2f}%)

2. CALIBRATION METRICS (ECE & NLL):
- Unimodal ECE (Joint Tuple):      {ece_uni_all:.2f}% (RGB: {ece_uni_rgb:.2f}%, IR: {ece_uni_ir:.2f}%) | NLL: {nll_uni_all:.4f}
- Bimodal Temp ECE:                {ece_bim_temp_all:.2f}% (RGB: {ece_bim_temp_rgb:.2f}%, IR: {ece_bim_temp_ir:.2f}%) | NLL: {nll_bim_temp_all:.4f}
- Bimodal Full ECE:                {ece_bim_full_all:.2f}% (RGB: {ece_bim_full_rgb:.2f}%, IR: {ece_bim_full_ir:.2f}%) | NLL: {nll_bim_full_all:.4f}

3. SAMPLES RESCUED:
- Total Rescued (Bimodal Full):    {rescued_bim_full} (Broken: {broken_bim_full})
- IR Rescued Samples:              {rescued_ir_full} / {ir_test_mask.sum()}
- RGB Rescued Samples:             {rescued_rgb_full} / {rgb_test_mask.sum()}

4. CALIBRATION PARAMETERS:
- w_RGB: {bim_hcdc.hcd_rgb.w.round(4).tolist()} | T_RGB: {bim_hcdc.hcd_rgb.T:.4f}
- w_IR:  {bim_hcdc.hcd_ir.w.round(4).tolist()} | T_IR:  {bim_hcdc.hcd_ir.T:.4f}
"""

    # Build structured metrics dict
    result_metrics = {
        "marginal_acc": marginal_mean_acc,
        "acc_type": acc_type_bim,
        "acc_make": acc_make_bim,
        "acc_model": acc_model_bim,
        "acc_exact_match": acc_exact_bim_full,
        "pct_invalid_total": 0.0,
        "pct_invalid_make_model": 0.0,
        "pct_invalid_model_type": 0.0,
        "rgb": {"acc_exact": acc_rgb_bim_full},
        "ir": {"acc_exact": acc_ir_bim_full},
        # Diagnostic details
        "marginal": {
            "exact": acc_exact_marg,
            "rgb": acc_rgb_marg,
            "ir": acc_ir_marg,
            "invalid_pct": invalid_pct_marg,
        },
        "hcd_pure": {
            "exact": acc_exact_pure,
            "rgb": acc_rgb_pure,
            "ir": acc_ir_pure,
            "rescued": diag_pure["rescued"],
        },
        "hcd_unimodal": {
            "exact": acc_exact_uni,
            "rgb": acc_rgb_uni,
            "ir": acc_ir_uni,
            "ece": ece_uni_all,
            "nll": nll_uni_all,
            "rescued": diag_uni["rescued"],
            "weights": hcdc_uni.w.tolist(),
            "temperature": float(hcdc_uni.T),
        },
        "hcd_bimodal_temp": {
            "exact": acc_exact_uni,
            "t_rgb": float(t_rgb_temp),
            "t_ir": float(t_ir_temp),
            "ece": ece_bim_temp_all,
            "nll": nll_bim_temp_all,
        },
        "hcd_bimodal_full": {
            "exact": acc_exact_bim_full,
            "rgb": acc_rgb_bim_full,
            "ir": acc_ir_bim_full,
            "ece": ece_bim_full_all,
            "nll": nll_bim_full_all,
            "rescued": rescued_bim_full,
            "broken": broken_bim_full,
            "rescued_ir": rescued_ir_full,
            "rescued_rgb": rescued_rgb_full,
            "weights_rgb": bim_hcdc.hcd_rgb.w.tolist(),
            "temp_rgb": float(bim_hcdc.hcd_rgb.T),
            "weights_ir": bim_hcdc.hcd_ir.w.tolist(),
            "temp_ir": float(bim_hcdc.hcd_ir.T),
        },
        "hcd_task_temperature": {
            "exact": acc_exact_task_scaled,
            "rgb": acc_rgb_task_scaled,
            "ir": acc_ir_task_scaled,
            "temps_rgb": task_temp_rgb,
            "temps_ir": task_temp_ir,
        },
    }

    return {
        "model_name": name,
        "backbone": backbone,
        "fold": fold,
        "metrics": result_metrics,
        "report_text": report_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bimodal Daylight/Infrared Temperature & Task Calibration in HCD-C"
    )
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Path to UFPR-VeSV root directory")
    parser.add_argument("--fold", type=int, default=0, help="Evaluation fold (0 to 9)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_ft", help="Checkpoints base directory")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "convnext", "convnext_base", "efficientnet", "efficientnet_v2", "swin", "swin_base"],
        help="Model architecture to evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--cache_dir", type=str, default="./extracted_logits", help="Directory to cache extracted logits")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save experiment results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    output_dir = Path(args.output_dir).resolve()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[*] Starting Bimodal Calibration Evaluation on Device: {device}")
    print(f"[*] Dataset: {data_dir} | Fold: {args.fold}")

    # Build taxonomy and catalog
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")
    catalog = Catalog(
        taxonomy.valid_tuples_tensor.numpy(),
        task_names=["type", "make", "model"],
        n_classes=[14, 26, 136],
    )
    print(f"[*] Valid Tuple Catalog loaded: {catalog}")

    available_models = [
        (
            "ConvNeXt-Base + LeJEPA",
            "convnext_base",
            checkpoints_dir / "convnext_lejepa_ft" / f"best_model_fold{args.fold}.pth",
        ),
        (
            "EfficientNet-V2 + LeJEPA",
            "tf_efficientnetv2_m.in21k_ft_in1k",
            checkpoints_dir / "efficientnet_v2_lejepa_ft" / f"best_model_fold{args.fold}.pth",
        ),
        (
            "Swin-Base + LeJEPA",
            "swin_base_patch4_window7_224",
            checkpoints_dir / "swin_t_lejepa_ft" / f"best_model_fold{args.fold}.pth",
        ),
    ]

    # Filter based on --model arg
    if args.model in ["convnext", "convnext_base"]:
        models_to_run = [available_models[0]]
    elif args.model in ["efficientnet", "efficientnet_v2"]:
        models_to_run = [available_models[1]]
    elif args.model in ["swin", "swin_base"]:
        models_to_run = [available_models[2]]
    else:
        models_to_run = available_models

    all_eval_results = []
    for name, backbone, ckpt_path in models_to_run:
        if not ckpt_path.exists():
            print(f"[-] Checkpoint {ckpt_path} not found. Skipping...")
            continue

        res = evaluate_backbone_bimodal(
            name=name,
            backbone=backbone,
            ckpt_path=ckpt_path,
            data_dir=data_dir,
            fold=args.fold,
            catalog=catalog,
            device=device,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_eval_results.append(res)

        # Save structured experiment folder and update results/benchmark_summary.csv
        exp_name = f"eval_hcdc_bimodal_{backbone}_fold{args.fold}"
        config_dict = {
            "experiment": exp_name,
            "backbone": backbone,
            "split_fold": args.fold,
            "data_dir": str(data_dir),
            "ckpt_path": str(ckpt_path),
            "device": str(device),
            "method": "bimodal_hcdc_calibration",
            "epochs": "30",
        }
        save_experiment_result(
            experiment_name=exp_name,
            config=config_dict,
            metrics=res["metrics"],
            report_text=res["report_text"],
            output_root=str(output_dir),
        )

    # Consolidated Multi-Model Table
    print("\n" + "=" * 105)
    print("CONSOLIDATED SUMMARY: BIMODAL CALIBRATION OF HCD-C ACROSS BACKBONES")
    print("=" * 105)
    print(
        f"{'Model':<25} | {'Marginal':<9} | {'HCD-C Uni':<10} | {'Bimodal Full':<12} | "
        f"{'IR (Uni -> Bim)':<16} | {'Rescued':<8} | {'ECE Bim'}"
    )
    print("-" * 105)
    for r in all_eval_results:
        m = r["metrics"]["marginal"]
        u = r["metrics"]["hcd_unimodal"]
        b = r["metrics"]["hcd_bimodal_full"]
        print(
            f"{r['model_name']:<25} | {m['exact']:>8.2f}% | {u['exact']:>9.2f}% | {b['exact']:>11.2f}% | "
            f"{u['ir']:>6.2f}% -> {b['ir']:>6.2f}% | {b['rescued']:>7} | {b['ece']:>6.2f}%"
        )
    print("=" * 105)

    # Save JSON summary of bimodal results
    bimodal_json_path = output_dir / f"hcdc_bimodal_results_fold{args.fold}.json"
    with open(bimodal_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {r["backbone"]: r["metrics"] for r in all_eval_results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[✓] Consolidated JSON saved to: {bimodal_json_path}")


if __name__ == "__main__":
    main()
