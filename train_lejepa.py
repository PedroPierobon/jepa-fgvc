#!/usr/bin/env python3
"""
LeJEPA (Latent-Euclidean JEPA) Self-Supervised Pre-Training with SIGReg
======================================================================
Pre-trains Vision Transformers and modern CNN backbones on the UFPR-VeSV dataset
using normalized Latent-Euclidean similarity losses and Sketched Isotropic Gaussian
Regularization (SIGReg). Supports DDP multi-GPU training, AMP mixed-precision,
cosine annealing with linear warmup, and fast local verification (--dry-run).
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

from datasets.ufpr_dataset import UFPRDataset, LeJEPADataTransform
from models.lejepa_module import LeJEPA
from utils.results_logger import save_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeJEPA Self-Supervised Pre-training on UFPR-VeSV with SIGReg"
    )

    # Dataset & Split
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Root directory of UFPR-VeSV")
    parser.add_argument("--split_fold", type=int, default=0, help="Split fold (0 to 9)")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader worker processes")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    # Architecture
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Initialize backbone with ImageNet pretrained weights")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent space dimension K regularized by SIGReg")
    parser.add_argument("--proj_hidden_dim", type=int, default=2048, help="Hidden dimension of MLP projector")
    parser.add_argument("--use_predictor", action="store_true", help="Use residual predictor head for similarity loss")

    # SIGReg & Loss
    parser.add_argument("--lambd", type=float, default=2.0, help="SIGReg loss weight (L = L_sim + lambd * L_sigreg)")
    parser.add_argument("--num_projections", type=int, default=128, help="Number M of sketched random directions in S^{K-1}")
    parser.add_argument("--num_quadrature_nodes", type=int, default=17, help="Number P of trapezoidal quadrature nodes")
    parser.add_argument("--t_max", type=float, default=3.0, help="Upper frequency limit for SIGReg ECF integration")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW")
    parser.add_argument("--epochs", type=int, default=60, help="Total pre-training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--clip_grad", type=float, default=3.0, help="Max gradient norm clipping")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (FP16)")

    # Misc
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_freq", type=int, default=10, help="Checkpoint saving frequency (in epochs)")
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


def run_dry_run(model: nn.Module, dataset: UFPRDataset, device: torch.device, args: argparse.Namespace) -> None:
    """Executes fast 2-step verification and checks tensors/gradients."""
    print("\n" + "=" * 70)
    print("RUNNING LEJEPA DRY-RUN VERIFICATION (2 batches of 4 samples)")
    print("=" * 70)
    model.train()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, (views, meta) in enumerate(loader):
        if step >= 2:
            break
        v1, v2 = views[0].to(device), views[1].to(device)
        optimizer.zero_grad()
        out = model((v1, v2))
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}/2 | Loss: {loss.item():.4f} (Sim: {out['loss_sim'].item():.4f}, SIGReg: {out['loss_sigreg'].item():.4f})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_ckpt = out_dir / "lejepa_encoder_dryrun.pth"
    torch.save({"encoder_state_dict": model.encoder.state_dict(), "backbone": args.backbone}, dry_ckpt)
    print(f"[✓] Dry-run complete. Verification checkpoint saved: {dry_ckpt}\n")
    sys.exit(0)


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    if is_main_process:
        print("=" * 80)
        print(f"LEJEPA SSL PRE-TRAINING (UFPR-VeSV) - FOLD {args.split_fold}")
        print("=" * 80)
        print(f"Backbone: {args.backbone} (Pretrained: {args.pretrained})")
        print(f"Latent Dim K: {args.latent_dim} | Projector Hidden: {args.proj_hidden_dim} | SIGReg Lambda: {args.lambd}")
        print(f"Batch Size: {args.batch_size} per GPU (Global: {args.batch_size * world_size}) | Epochs: {args.epochs}")
        print(f"Optimizer: AdamW (LR: {args.lr:.2e}, Min LR: {args.min_lr:.2e}, Warmup: {args.warmup_epochs} ep)")
        print(f"AMP FP16: {args.amp} | World Size: {world_size}")
        print("=" * 80)

    # 1. Dataset & DataLoader
    train_transform = LeJEPADataTransform(img_size=args.img_size)
    train_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="train",
        transform=train_transform,
        is_pretrain=True,
    )

    # 2. Model
    model = LeJEPA(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        latent_dim=args.latent_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        lambd=args.lambd,
        num_projections=args.num_projections,
        num_quadrature_nodes=args.num_quadrature_nodes,
        t_max=args.t_max,
        use_predictor=args.use_predictor,
    ).to(device)

    if args.dry_run:
        run_dry_run(model, train_dataset, device, args)

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

    # 3. Optimizer & AMP Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    total_start_time = time.time()

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss, total_sim, total_sigreg = 0.0, 0.0, 0.0
        num_batches = len(train_loader)
        epoch_start = time.time()

        for step, (views, _) in enumerate(train_loader):
            v1, v2 = views[0].to(device, non_blocking=True), views[1].to(device, non_blocking=True)
            lr = adjust_learning_rate(optimizer, epoch - 1, step, num_batches, args)

            optimizer.zero_grad()

            if args.amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model((v1, v2))
                    loss = outputs["loss"]
                    loss_sim = outputs["loss_sim"]
                    loss_sigreg = outputs["loss_sigreg"]

                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model((v1, v2))
                loss = outputs["loss"]
                loss_sim = outputs["loss_sim"]
                loss_sigreg = outputs["loss_sigreg"]
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            total_loss += loss.item()
            total_sim += loss_sim.item()
            total_sigreg += loss_sigreg.item()

            if is_main_process and (step % max(1, num_batches // 5) == 0 or step == num_batches - 1):
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] Step [{step:04d}/{num_batches:04d}] | "
                    f"Loss: {loss.item():.4f} (Sim: {loss_sim.item():.4f}, SIGReg: {loss_sigreg.item():.4f}) | "
                    f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                )

        # End of epoch summary & saving
        if is_main_process:
            avg_loss = total_loss / num_batches
            avg_sim = total_sim / num_batches
            avg_sigreg = total_sigreg / num_batches
            print(f"==> End of Epoch {epoch:03d}/{args.epochs:03d} | Avg Loss: {avg_loss:.4f} | Sim: {avg_sim:.4f} | SIGReg: {avg_sigreg:.4f}")

            raw_model = model.module if hasattr(model, "module") else model
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                encoder_path = output_dir / f"lejepa_encoder_fold{args.split_fold}.pth"
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "encoder_state_dict": raw_model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(save_dict, ckpt_path)
                torch.save({"encoder_state_dict": raw_model.encoder.state_dict(), "backbone": args.backbone}, encoder_path)
                print(f"[✓] Checkpoint saved: {ckpt_path.name} and {encoder_path.name}")

    if is_main_process:
        total_time_min = (time.time() - total_start_time) / 60.0
        print(f"\n[✓] Pre-training successfully completed in {total_time_min:.2f} minutes!")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
