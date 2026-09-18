#!/usr/bin/env python3
"""
Prototype-Guided Vision-Language JEPA (VL-JEPA) Pre-Training with SIGReg
=======================================================================
Pre-trains visual encoders (e.g., ConvNeXt-Base) on UFPR-VeSV by aligning visual
representations directly with 225 frozen semantic text prototypes in Euclidean space.
Combines:
  1. Multimodal Euclidean Prototype Alignment (MSE without contrastive denominator)
  2. Latent-Euclidean Cross-View Invariance (LeJEPA view consistency)
  3. Sketched Isotropic Gaussian Regularization (SIGReg) preventing dimensional collapse
  4. Built-in Zero-Shot Evaluation (100% compliant with vehicle taxonomy, 0.00% invalid)
Supports DDP multi-GPU training, AMP mixed-precision, and fast local verification (--dry-run).
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets.ufpr_dataset import UFPRDataset, LeJEPADataTransform, get_eval_transform
from evaluate_hierarchical import VehicleTaxonomy, compute_hierarchical_metrics
from models.vl_jepa_module import VLJEPA, PrototypeTextBank
from utils.results_logger import save_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype-Guided VL-JEPA Pre-training on UFPR-VeSV with SIGReg"
    )

    # Dataset & Split
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Root directory of UFPR-VeSV")
    parser.add_argument("--split_fold", type=int, default=0, help="Split fold (0 to 9)")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader worker processes")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vl_jepa_convnext_prototypes", help="Directory to save checkpoints")

    # Architecture & Vision Backbone
    parser.add_argument("--backbone", type=str, default="convnext_base", help="Backbone model architecture")
    parser.add_argument("--pretrained", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Initialize backbone with ImageNet pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze visual encoder backbone (train projector only)")
    parser.add_argument("--proj_hidden_dim", type=int, default=2048, help="Hidden dimension of MLP projector")
    parser.add_argument("--use_predictor", action="store_true", help="Use residual predictor head for cross-view similarity loss")

    # Text Model & Prototypes
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence Transformer model for text prototypes")
    parser.add_argument("--prompt_template", type=str, default="A photo of a {type}, make {make}, model {model}.", help="Template for textual prototype descriptions")

    # Loss Components & SIGReg
    parser.add_argument("--alpha_proto", type=float, default=1.0, help="Weight of prototype Euclidean alignment loss")
    parser.add_argument("--beta_sim", type=float, default=1.0, help="Weight of cross-view Latent-Euclidean similarity loss")
    parser.add_argument("--lambd", type=float, default=2.0, help="Weight of SIGReg loss (lambda)")
    parser.add_argument("--num_projections", type=int, default=128, help="Number M of sketched random directions in S^{K-1}")
    parser.add_argument("--num_quadrature_nodes", type=int, default=17, help="Number P of trapezoidal quadrature nodes")
    parser.add_argument("--t_max", type=float, default=3.0, help="Upper frequency limit for SIGReg ECF integration")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1.8e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW")
    parser.add_argument("--epochs", type=int, default=60, help="Total pre-training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--clip_grad", type=float, default=3.0, help="Max gradient norm clipping")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (FP16)")

    # Data Augmentation & Spectral IR Simulation
    parser.add_argument("--spectral_ir_aug", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Enable spectral infrared simulation (CLAHE + sensor noise)")
    parser.add_argument("--grayscale_prob", type=float, default=0.40, help="Probability of random grayscale transform")

    # Evaluation & Misc
    parser.add_argument("--eval_freq", type=int, default=5, help="Frequency of Zero-Shot validation (in epochs)")
    parser.add_argument("--save_freq", type=int, default=10, help="Checkpoint saving frequency (in epochs)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Quick local verification on 2 batches of 4 samples")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        is_distributed = False
        rank, local_rank, world_size = 0, 0, 1
    return is_distributed, rank, local_rank, world_size


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    args: argparse.Namespace,
) -> float:
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step

    if current_step < warmup_steps:
        lr = args.lr * (current_step + 1) / max(1, warmup_steps)
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@torch.no_grad()
def evaluate_zero_shot(
    model: nn.Module,
    val_loader: DataLoader,
    taxonomy: VehicleTaxonomy,
    device: torch.device,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    Evaluates zero-shot taxonomic vehicle classification by projecting visual features
    and querying the 225 text prototypes.
    """
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model

    all_pred_type, all_pred_make, all_pred_model = [], [], []
    all_true_type, all_true_make, all_true_model = [], [], []
    all_is_ir = []

    for images, meta in val_loader:
        images = images.to(device, non_blocking=True)
        z = raw_model.forward_projector(images)
        _, p_t, p_m, p_mo = raw_model.prototype_bank.classify_zero_shot(z, metric=metric)

        all_pred_type.append(p_t.cpu())
        all_pred_make.append(p_m.cpu())
        all_pred_model.append(p_mo.cpu())

        all_true_type.append(meta["target_type"])
        all_true_make.append(meta["target_make"])
        all_true_model.append(meta["target_model"])
        all_is_ir.append(meta["is_infrared"])

    pred_t = torch.cat(all_pred_type, dim=0)
    pred_m = torch.cat(all_pred_make, dim=0)
    pred_mo = torch.cat(all_pred_model, dim=0)

    true_t = torch.cat(all_true_type, dim=0)
    true_m = torch.cat(all_true_make, dim=0)
    true_mo = torch.cat(all_true_model, dim=0)
    is_ir = torch.cat(all_is_ir, dim=0)

    metrics = compute_hierarchical_metrics(
        pred_type=pred_t,
        pred_make=pred_m,
        pred_model=pred_mo,
        true_type=true_t,
        true_make=true_m,
        true_model=true_mo,
        is_infrared=is_ir,
        taxonomy=taxonomy,
        prefix="zero_shot",
    )
    return metrics


def run_dry_run(
    model: nn.Module,
    train_dataset: UFPRDataset,
    val_dataset: UFPRDataset,
    taxonomy: VehicleTaxonomy,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Executes fast 2-step verification and checks tensors/gradients/metrics."""
    print("\n" + "=" * 70)
    print("RUNNING PROTOTYPE-GUIDED VL-JEPA DRY-RUN VERIFICATION (2 batches of 4)")
    print("=" * 70)
    model.train()
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    raw_model = model.module if hasattr(model, "module") else model

    for step, (views, meta) in enumerate(loader):
        if step >= 2:
            break
        v1, v2 = views[0].to(device), views[1].to(device)
        y_t = meta["target_type"].to(device)
        y_m = meta["target_make"].to(device)
        y_mo = meta["target_model"].to(device)

        optimizer.zero_grad()
        out = raw_model.forward_train(
            v1, v2, y_t, y_m, y_mo,
            alpha_proto=args.alpha_proto,
            beta_sim=args.beta_sim,
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        print(
            f"Step {step+1}/2 | Loss: {loss.item():.4f} "
            f"(Proto: {out['loss_proto'].item():.4f}, Sim: {out['loss_sim'].item():.4f}, SIGReg: {out['loss_sigreg'].item():.4f})"
        )

    # Fast validation test (4 samples)
    print("\n[*] Testing zero-shot validation routine...")
    from torch.utils.data import Subset
    val_loader = DataLoader(Subset(val_dataset, list(range(min(4, len(val_dataset))))), batch_size=4, shuffle=False)
    metrics = evaluate_zero_shot(model, val_loader, taxonomy, device)
    print(f"Zero-Shot Exact Tuple: {metrics['acc_exact_match']:.2f}% | Marginal: {metrics['marginal_acc']:.2f}%")
    print(f"Invalid Tuples: {metrics['pct_invalid_total']:.2f}% (Must be 0.00% by construction)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_ckpt = out_dir / "vl_jepa_encoder_dryrun.pth"
    torch.save({"encoder_state_dict": raw_model.encoder.state_dict(), "backbone": args.backbone}, dry_ckpt)
    print(f"[✓] Dry-run complete. Verification checkpoint saved: {dry_ckpt}\n")
    sys.exit(0)


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"PROTOTYPE-GUIDED VL-JEPA PRE-TRAINING (UFPR-VeSV) - FOLD {args.split_fold}")
        print("=" * 80)
        print(f"Backbone: {args.backbone} (Pretrained: {args.pretrained})")
        print(f"Text Model: {args.text_model} | Prompt: \"{args.prompt_template}\"")
        print(f"Loss Weights -> Alpha (Proto): {args.alpha_proto} | Beta (Sim): {args.beta_sim} | Lambda (SIGReg): {args.lambd}")
        print(f"Batch Size: {args.batch_size} per GPU (Global: {args.batch_size * world_size}) | Epochs: {args.epochs}")
        print(f"Optimizer: AdamW (LR: {args.lr:.2e}, Min LR: {args.min_lr:.2e}, Warmup: {args.warmup_epochs} ep)")
        print(f"Spectral IR Aug: {args.spectral_ir_aug} | Grayscale Prob: {args.grayscale_prob}")
        print(f"AMP FP16: {args.amp} | World Size: {world_size}")
        print("=" * 80)

    # 1. Image Resolution and Normalization
    is_pe_core = ("PE-Core" in args.backbone) or ("pe_core" in args.backbone)
    if is_pe_core and args.img_size == 224:
        args.img_size = 336
        if is_main_process:
            print("[*] Automatically setting img_size=336 for PE-Core backbone.")

    norm_mean = (0.5, 0.5, 0.5) if is_pe_core else (0.485, 0.456, 0.406)
    norm_std = (0.5, 0.5, 0.5) if is_pe_core else (0.229, 0.224, 0.225)

    # Dataset & DataLoader
    train_transform = LeJEPADataTransform(
        img_size=args.img_size,
        mean=norm_mean,
        std=norm_std,
        grayscale_prob=args.grayscale_prob,
        spectral_ir_aug=args.spectral_ir_aug,
    )
    train_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="train",
        transform=train_transform,
        is_pretrain=True,
    )

    eval_transform = get_eval_transform(
        img_size=args.img_size,
        mean=norm_mean,
        std=norm_std,
    )
    val_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="val",
        transform=eval_transform,
        is_pretrain=False,
    )

    annotations_file = Path(args.data_dir) / "annotations.json"
    taxonomy = VehicleTaxonomy(annotations_file)

    # 2. Text Prototypes Cache Management
    cache_prototypes_path = output_dir / f"text_prototypes_bank_{Path(args.text_model).name}.pt"
    if is_distributed:
        if is_main_process:
            # Rank 0 builds and caches prototypes first to avoid concurrent downloads
            _ = PrototypeTextBank(
                annotations_file=annotations_file,
                text_model_name=args.text_model,
                prompt_template=args.prompt_template,
                cache_path=cache_prototypes_path,
            )
        dist.barrier()

    # 3. Instantiate VL-JEPA Model
    model = VLJEPA(
        annotations_file=annotations_file,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        text_model_name=args.text_model,
        proj_hidden_dim=args.proj_hidden_dim,
        use_predictor=args.use_predictor,
        freeze_backbone=args.freeze_backbone,
        lambd_sigreg=args.lambd,
        num_projections=args.num_projections,
        num_quadrature_nodes=args.num_quadrature_nodes,
        t_max=args.t_max,
        cache_prototypes_path=cache_prototypes_path,
    ).to(device)

    if args.dry_run:
        run_dry_run(model, train_dataset, val_dataset, taxonomy, device, args)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Optimizer & AMP Scaler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main_process:
        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"[*] Trainable parameters: {num_trainable:,} / {num_total:,} ({100.0 * num_trainable / num_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    start_epoch = 1
    total_start_time = time.time()
    best_exact_acc = 0.0
    best_metrics = {}

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss, total_proto, total_sim, total_sigreg = 0.0, 0.0, 0.0, 0.0
        num_batches = len(train_loader)
        epoch_start = time.time()

        for step, (views, meta) in enumerate(train_loader):
            v1 = views[0].to(device, non_blocking=True)
            v2 = views[1].to(device, non_blocking=True)
            y_t = meta["target_type"].to(device, non_blocking=True)
            y_m = meta["target_make"].to(device, non_blocking=True)
            y_mo = meta["target_model"].to(device, non_blocking=True)

            lr = adjust_learning_rate(optimizer, epoch - 1, step, num_batches, args)
            optimizer.zero_grad()

            raw_model = model.module if hasattr(model, "module") else model

            if args.amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = raw_model.forward_train(
                        v1, v2, y_t, y_m, y_mo,
                        alpha_proto=args.alpha_proto,
                        beta_sim=args.beta_sim,
                    )
                    loss = outputs["loss"]
                    loss_proto = outputs["loss_proto"]
                    loss_sim = outputs["loss_sim"]
                    loss_sigreg = outputs["loss_sigreg"]

                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = raw_model.forward_train(
                    v1, v2, y_t, y_m, y_mo,
                    alpha_proto=args.alpha_proto,
                    beta_sim=args.beta_sim,
                )
                loss = outputs["loss"]
                loss_proto = outputs["loss_proto"]
                loss_sim = outputs["loss_sim"]
                loss_sigreg = outputs["loss_sigreg"]

                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            total_loss += loss.item()
            total_proto += loss_proto.item()
            total_sim += loss_sim.item()
            total_sigreg += loss_sigreg.item()

            if is_main_process and (step % max(1, num_batches // 5) == 0 or step == num_batches - 1):
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] Step [{step:04d}/{num_batches:04d}] | "
                    f"Loss: {loss.item():.4f} (Proto: {loss_proto.item():.4f}, Sim: {loss_sim.item():.4f}, SIGReg: {loss_sigreg.item():.4f}) | "
                    f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                )

        # End of Epoch Summary
        if is_main_process:
            avg_loss = total_loss / num_batches
            avg_proto = total_proto / num_batches
            avg_sim = total_sim / num_batches
            avg_sigreg = total_sigreg / num_batches
            print(
                f"==> End of Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Avg Loss: {avg_loss:.4f} | Proto: {avg_proto:.4f} | Sim: {avg_sim:.4f} | SIGReg: {avg_sigreg:.4f}"
            )

            # Checkpoint saving
            raw_model = model.module if hasattr(model, "module") else model
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                encoder_path = output_dir / f"vl_jepa_encoder_fold{args.split_fold}.pth"
                torch.save(
                    {"encoder_state_dict": raw_model.encoder.state_dict(), "backbone": args.backbone},
                    encoder_path,
                )
                last_path = output_dir / "checkpoint_last.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "encoder_state_dict": raw_model.encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    last_path,
                )
                print(f"[✓] Checkpoint saved: {encoder_path.name}")

            # Zero-Shot Validation Evaluation
            if epoch % args.eval_freq == 0 or epoch == args.epochs:
                val_start = time.time()
                val_metrics = evaluate_zero_shot(model, val_loader, taxonomy, device)
                val_time = time.time() - val_start

                acc_exact = val_metrics["acc_exact_match"]
                marginal = val_metrics["marginal_acc"]
                t_acc = val_metrics["acc_type"]
                m_acc = val_metrics["acc_make"]
                mo_acc = val_metrics["acc_model"]
                rgb_acc = val_metrics["rgb"]["acc_exact"]
                ir_acc = val_metrics["ir"]["acc_exact"]
                inv_pct = val_metrics["pct_invalid_total"]

                print("\n" + "-" * 75)
                print(f"ZERO-SHOT EVALUATION AT EPOCH {epoch:03d} ({val_time:.1f}s):")
                print(f"  Exact Tuple Acc:     {acc_exact:6.2f}%  (Invalid Tuples: {inv_pct:4.2f}%)")
                print(f"  Marginal Acc:        {marginal:6.2f}%  [Type: {t_acc:.2f}%, Make: {m_acc:.2f}%, Model: {mo_acc:.2f}%]")
                print(f"  Daylight (RGB) Exact:{rgb_acc:6.2f}%  |  Infrared (IR) Exact: {ir_acc:6.2f}%")
                print("-" * 75 + "\n")

                if acc_exact > best_exact_acc:
                    best_exact_acc = acc_exact
                    best_metrics = val_metrics
                    best_path = output_dir / "checkpoint_best.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "encoder_state_dict": raw_model.encoder.state_dict(),
                            "metrics": val_metrics,
                            "args": vars(args),
                        },
                        best_path,
                    )
                    print(f"[★] New best zero-shot exact accuracy: {best_exact_acc:.2f}%! Saved {best_path.name}\n")

    # 6. Final Test Split Evaluation & Experiment Logging
    if is_main_process:
        total_time_min = (time.time() - total_start_time) / 60.0
        final_metrics = best_metrics if best_metrics else val_metrics

        print("\n" + "=" * 80)
        print(f"RUNNING FINAL TEST SPLIT EVALUATION (FOLD {args.split_fold})")
        print("=" * 80)
        test_dataset = UFPRDataset(
            root_dir=args.data_dir,
            split_fold=args.split_fold,
            subset="test",
            transform=eval_transform,
            is_pretrain=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        raw_model = model.module if hasattr(model, "module") else model
        best_path = output_dir / "checkpoint_best.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[*] Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} for test evaluation.")

        test_metrics = evaluate_zero_shot(model, test_loader, taxonomy, device)
        print(f"  Test Exact Tuple Acc:      {test_metrics['acc_exact_match']:6.2f}% (Invalid: {test_metrics['pct_invalid_total']:4.2f}%)")
        print(f"  Test Marginal Acc:         {test_metrics['marginal_acc']:6.2f}%")
        print(f"  Test Daylight (RGB) Exact: {test_metrics['rgb']['acc_exact']:6.2f}%")
        print(f"  Test Infrared (IR) Exact:  {test_metrics['ir']['acc_exact']:6.2f}%")
        print("=" * 80 + "\n")

        report_text = f"""================================================================================
PROTOTYPE-GUIDED VL-JEPA EVALUATION REPORT
================================================================================
Dataset: UFPR-VeSV | Fold: {args.split_fold} | Backbone: {args.backbone}
Text Model: {args.text_model}
Epochs: {args.epochs} | Training Time: {total_time_min:.2f} minutes
Final Loss: {avg_loss:.4f} | Proto Loss: {avg_proto:.4f} | Sim: {avg_sim:.4f} | SIGReg: {avg_sigreg:.4f}
Spectral IR Augmentation: {args.spectral_ir_aug} | Grayscale Prob: {args.grayscale_prob}

VAL ZERO-SHOT TAXONOMIC PERFORMANCE (Valid Catalog = 225 Tuples):
--------------------------------------------------------------------------------
Exact-Match Full-Tuple Accuracy:  {final_metrics.get('acc_exact_match', 0.0):6.2f}%
Marginal Accuracy (Average):       {final_metrics.get('marginal_acc', 0.0):6.2f}%
  - Body Type Accuracy (14 cls):  {final_metrics.get('acc_type', 0.0):6.2f}%
  - Vehicle Make Accuracy (26 cls):{final_metrics.get('acc_make', 0.0):6.2f}%
  - Vehicle Model Accuracy (136): {final_metrics.get('acc_model', 0.0):6.2f}%
Daylight (Visible / RGB) Exact:   {final_metrics.get('rgb', {}).get('acc_exact', 0.0):6.2f}%
Infrared (Nighttime / IR) Exact:  {final_metrics.get('ir', {}).get('acc_exact', 0.0):6.2f}%

TEST ZERO-SHOT TAXONOMIC PERFORMANCE (OFFICIAL TEST SPLIT):
--------------------------------------------------------------------------------
Test Exact-Match Full-Tuple Acc:  {test_metrics.get('acc_exact_match', 0.0):6.2f}%
Test Marginal Accuracy (Average): {test_metrics.get('marginal_acc', 0.0):6.2f}%
Test Daylight (RGB) Exact:        {test_metrics.get('rgb', {}).get('acc_exact', 0.0):6.2f}%
Test Infrared (IR) Exact:         {test_metrics.get('ir', {}).get('acc_exact', 0.0):6.2f}%
Invalid Full Tuples:              {test_metrics.get('pct_invalid_total', 0.0):6.2f}% (Enforced 0.00% by design)
================================================================================
"""
        final_metrics_dict = dict(final_metrics)
        final_metrics_dict.update({
            "loss": avg_loss,
            "loss_proto": avg_proto,
            "loss_sim": avg_sim,
            "loss_sigreg": avg_sigreg,
            "training_time_min": total_time_min,
            "epochs": args.epochs,
            "spectral_ir_aug": args.spectral_ir_aug,
            "test": test_metrics,
        })

        exp_name = f"vl_jepa_{args.backbone}_prototypes_fold{args.split_fold}"
        save_experiment_result(
            experiment_name=exp_name,
            config=vars(args),
            metrics=final_metrics_dict,
            report_text=report_text,
            output_root="./results",
        )
        print(f"\n[✓] Prototype-Guided VL-JEPA pre-training completed in {total_time_min:.2f} minutes!")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
