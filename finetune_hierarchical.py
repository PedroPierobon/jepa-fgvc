#!/usr/bin/env python3
"""
End-to-End Supervised Fine-Tuning & Benchmarking (UFPR-VeSV)
===========================================================
Trains Vision Transformers and modern CNN backbones end-to-end for fine-grained
vehicle categorization (Type: 14, Make: 26, Model: 136).
Supports three scientific regimes:
1. Direct Fine-Tuning from ImageNet pretrained weights (baseline).
2. Fine-Tuning initialized from LeJEPA + SIGReg self-supervised checkpoints.
3. Training from scratch (random initialization).
Automatically logs configs, metrics, and summary rows to results/benchmark_summary.csv.
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
import torchvision.transforms as transforms

from datasets.ufpr_dataset import UFPRDataset, get_eval_transform
from models.lejepa_module import LeJEPAEncoder
from evaluate_hierarchical import (
    VehicleTaxonomy,
    compute_hierarchical_metrics,
    format_marginal_report,
    print_marginal_report,
)
from utils.results_logger import save_experiment_result


class ModelEMA:
    """
    Maintains Exponential Moving Average (EMA) of model parameters for improved
    generalization and calibration during extended fine-tuning schedules.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ) -> None:
        import copy
        self.decay = decay
        self.device = device
        raw_model = model.module if hasattr(model, "module") else model
        self.module = copy.deepcopy(raw_model)
        self.module.eval()
        if device is not None:
            self.module.to(device)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        for ema_p, model_p in zip(self.module.parameters(), raw_model.parameters()):
            if ema_p.dtype.is_floating_point:
                ema_p.data.mul_(self.decay).add_(
                    model_p.data.to(ema_p.device), alpha=1.0 - self.decay
                )
        for ema_b, model_b in zip(self.module.buffers(), raw_model.buffers()):
            ema_b.copy_(model_b.to(ema_b.device))


def get_layer_id_and_scale(
    name: str, backbone_name: str, decay_rate: float = 0.75
) -> Tuple[int, float]:
    """
    Assigns layer ID and learning rate decay scale for Layer-wise Learning Rate Decay (LLRD).
    Earlier layers receive smaller learning rates to preserve general representations.
    """
    name_lower = name.lower()
    b_lower = backbone_name.lower()

    if "convnext" in b_lower:
        max_layer = 4
        if "stem" in name_lower:
            layer_id = 0
        elif "stages.0" in name_lower or "stages_0" in name_lower:
            layer_id = 1
        elif "stages.1" in name_lower or "stages_1" in name_lower:
            layer_id = 2
        elif "stages.2" in name_lower or "stages_2" in name_lower:
            layer_id = 3
        elif "stages.3" in name_lower or "stages_3" in name_lower:
            layer_id = 4
        else:
            layer_id = 4
    elif "swin" in b_lower:
        max_layer = 4
        if "patch_embed" in name_lower:
            layer_id = 0
        elif "layers.0" in name_lower or "layers_0" in name_lower:
            layer_id = 1
        elif "layers.1" in name_lower or "layers_1" in name_lower:
            layer_id = 2
        elif "layers.2" in name_lower or "layers_2" in name_lower:
            layer_id = 3
        elif "layers.3" in name_lower or "layers_3" in name_lower:
            layer_id = 4
        else:
            layer_id = 4
    elif "vit" in b_lower:
        max_layer = 12
        if "patch_embed" in name_lower or "cls_token" in name_lower or "pos_embed" in name_lower:
            layer_id = 0
        else:
            import re
            m = re.search(r"blocks\.(\d+)", name_lower)
            if m:
                layer_id = min(int(m.group(1)) + 1, max_layer)
            else:
                layer_id = max_layer
    elif "efficientnet" in b_lower:
        max_layer = 7
        if "conv_stem" in name_lower or "bn1" in name_lower:
            layer_id = 0
        else:
            import re
            m = re.search(r"blocks\.(\d+)", name_lower)
            if m:
                layer_id = min(int(m.group(1)) + 1, max_layer)
            else:
                layer_id = max_layer
    else:
        max_layer = 1
        layer_id = 1

    scale = float(decay_rate ** (max_layer - layer_id))
    return layer_id, scale


class HierarchicalClassifier(nn.Module):
    """
    End-to-End Hierarchical Classification Model:
    Backbone (Encoder) + 3 Decision Heads (Type: 14, Make: 26, Model: 136).
    """

    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_m.in21k_ft_in1k",
        pretrained: bool = True,
        checkpoint_encoder_path: str = "",
        dropout: float = 0.2,
        num_types: int = 14,
        num_makes: int = 26,
        num_models: int = 136,
    ) -> None:
        super().__init__()
        self.encoder = LeJEPAEncoder(backbone_name=backbone_name, pretrained=pretrained)
        self.embed_dim = self.encoder.embed_dim

        # Load LeJEPA pre-trained encoder weights if provided
        if checkpoint_encoder_path and os.path.isfile(checkpoint_encoder_path):
            print(f"[*] Loading LeJEPA pretrained weights from: {checkpoint_encoder_path}")
            ckpt = torch.load(checkpoint_encoder_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
                self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                encoder_keys = {
                    k.replace("encoder.", "").replace("module.encoder.", ""): v
                    for k, v in ckpt["model_state_dict"].items()
                    if "encoder." in k
                }
                if encoder_keys:
                    self.encoder.load_state_dict(encoder_keys)
                else:
                    self.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                self.encoder.load_state_dict(ckpt, strict=False)
            print("[✓] Encoder successfully initialized from LeJEPA pre-trained representations!")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head_type = nn.Linear(self.embed_dim, num_types)
        self.head_make = nn.Linear(self.embed_dim, num_makes)
        self.head_model = nn.Linear(self.embed_dim, num_models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        feats = self.dropout(feats)
        return self.head_type(feats), self.head_make(feats), self.head_model(feats)


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Data augmentations for supervised fine-tuning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End Supervised Fine-Tuning & Evaluation on UFPR-VeSV"
    )

    # Dataset & Paths
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Root directory of dataset")
    parser.add_argument("--split_fold", type=int, default=0, help="Split fold index (0 to 9)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_finetune", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory where experiment logs/CSV are saved")
    parser.add_argument("--exp_name", type=str, default="", help="Custom experiment name for results tracking")
    parser.add_argument(
        "--lejepa_checkpoint",
        type=str,
        default="",
        help="Optional LeJEPA SSL checkpoint path to initialize encoder from",
    )

    # Architecture
    parser.add_argument(
        "--backbone",
        type=str,
        default="tf_efficientnetv2_m.in21k_ft_in1k",
        help="Model architecture name (e.g. tf_efficientnetv2_m.in21k_ft_in1k, swinv2_base_window12to16_192to256, convnext_base, resnet50)",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate before classifier heads")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr_backbone", type=float, default=3e-5, help="Learning rate for backbone")
    parser.add_argument("--lr_head", type=float, default=5e-4, help="Learning rate for classifier heads")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (FP16)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--loss_weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Taxonomy loss weights [w_type, w_make, w_model] (default: 1.0 1.0 1.0, e.g. 0.5 1.0 1.8)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (default: 0.0)",
    )
    # LLRD & EMA
    parser.add_argument(
        "--use_llrd",
        action="store_true",
        help="Enable Layer-wise Learning Rate Decay (LLRD) for backbone parameters",
    )
    parser.add_argument(
        "--llrd_decay",
        type=float,
        default=0.75,
        help="Multiplicative layer-wise learning rate decay rate (default: 0.75)",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Maintain Exponential Moving Average (EMA) of model parameters",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate (default: 0.999)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute fast 2-step verification and exit",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Input image resolution (alias for --img_size, e.g. 288)",
    )
    parser.add_argument(
        "--init_from_model",
        type=str,
        default="",
        help="Path to full HierarchicalClassifier checkpoint to warm-start progressive fine-tuning",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    if args.resolution > 0:
        args.img_size = args.resolution
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    taxonomy: VehicleTaxonomy,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates the full model and computes all taxonomic metrics."""
    model.eval()
    all_pred_type: List[torch.Tensor] = []
    all_pred_make: List[torch.Tensor] = []
    all_pred_model: List[torch.Tensor] = []
    all_true_type: List[torch.Tensor] = []
    all_true_make: List[torch.Tensor] = []
    all_true_model: List[torch.Tensor] = []
    all_is_ir: List[torch.Tensor] = []

    raw_model = model.module if hasattr(model, "module") else model

    for images, meta in loader:
        images = images.to(device, non_blocking=True)
        out_t, out_m, out_mo = raw_model(images)

        all_pred_type.append(out_t.argmax(dim=-1).cpu())
        all_pred_make.append(out_m.argmax(dim=-1).cpu())
        all_pred_model.append(out_mo.argmax(dim=-1).cpu())

        all_true_type.append(meta["target_type"].cpu())
        all_true_make.append(meta["target_make"].cpu())
        all_true_model.append(meta["target_model"].cpu())
        all_is_ir.append(meta["is_infrared"].cpu())

    pred_type = torch.cat(all_pred_type, dim=0)
    pred_make = torch.cat(all_pred_make, dim=0)
    pred_model = torch.cat(all_pred_model, dim=0)
    true_type = torch.cat(all_true_type, dim=0)
    true_make = torch.cat(all_true_make, dim=0)
    true_model = torch.cat(all_true_model, dim=0)
    is_ir = torch.cat(all_is_ir, dim=0)

    metrics = compute_hierarchical_metrics(
        pred_type=pred_type,
        pred_make=pred_make,
        pred_model=pred_model,
        true_type=true_type,
        true_make=true_make,
        true_model=true_model,
        is_infrared=is_ir,
        taxonomy=taxonomy,
        prefix="Fine-Tuning",
    )
    return metrics


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    data_dir = Path(args.data_dir)
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")

    if is_main_process:
        print("=" * 80)
        print(f"END-TO-END SUPERVISED FINE-TUNING - FOLD {args.split_fold}")
        print("=" * 80)
        print(f"Backbone: {args.backbone} | Resolution: {args.img_size}x{args.img_size}")
        print(f"Weight Origin: {'LeJEPA Checkpoint: ' + args.lejepa_checkpoint if args.lejepa_checkpoint else ('ImageNet Pretrained' if args.pretrained else 'From Scratch')}")
        print(f"LR Backbone: {args.lr_backbone:.2e} | LR Heads: {args.lr_head:.2e} | Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size} per GPU (Global: {args.batch_size * world_size}) | AMP: {args.amp}")
        print("=" * 80)

    # 1. Datasets & Loaders
    train_transform = get_train_transforms(img_size=args.img_size)
    eval_transform = get_eval_transform(img_size=args.img_size)

    train_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="train",
        transform=train_transform,
        is_pretrain=False,
    )
    val_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="val",
        transform=eval_transform,
        is_pretrain=False,
    )
    test_dataset = UFPRDataset(
        root_dir=args.data_dir,
        split_fold=args.split_fold,
        subset="test",
        transform=eval_transform,
        is_pretrain=False,
    )

    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2. Model
    model = HierarchicalClassifier(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        checkpoint_encoder_path=args.lejepa_checkpoint,
        dropout=args.dropout,
        num_types=taxonomy.num_types,
        num_makes=taxonomy.num_makes,
        num_models=taxonomy.num_models,
    ).to(device)

    # Warm-start full model (encoder + heads) if init_from_model is provided
    if args.init_from_model and os.path.isfile(args.init_from_model):
        if is_main_process:
            print(f"[*] Warm-starting full model (encoder + heads) from: {args.init_from_model}")
        full_ckpt = torch.load(args.init_from_model, map_location="cpu", weights_only=False)
        state = full_ckpt.get("ema_state_dict", full_ckpt.get("model_state_dict", full_ckpt))
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if is_main_process:
            print(f"[✓] Full model weights loaded successfully! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # 3. Differential Learning Rates & LLRD
    raw_model = model.module if hasattr(model, "module") else model
    if args.use_llrd:
        layer_params: Dict[int, List[nn.Parameter]] = {}
        layer_scales: Dict[int, float] = {}
        for n, p in raw_model.encoder.backbone.named_parameters():
            if not p.requires_grad:
                continue
            lid, sc = get_layer_id_and_scale(n, args.backbone, args.llrd_decay)
            layer_params.setdefault(lid, []).append(p)
            layer_scales[lid] = sc

        optimizer_grouped_parameters = []
        for lid in sorted(layer_params.keys()):
            scale = layer_scales[lid]
            lr_layer = args.lr_backbone * scale
            optimizer_grouped_parameters.append({
                "params": layer_params[lid],
                "lr": lr_layer,
                "weight_decay": args.weight_decay,
            })
            if is_main_process:
                print(f"[*] LLRD Stage {lid}: {len(layer_params[lid])} parameter tensors | LR: {lr_layer:.2e} (scale: {scale:.4f})")

        head_params = list(raw_model.head_type.parameters()) + list(raw_model.head_make.parameters()) + list(raw_model.head_model.parameters())
        optimizer_grouped_parameters.append({
            "params": head_params,
            "lr": args.lr_head,
            "weight_decay": args.weight_decay,
        })
    else:
        optimizer_grouped_parameters = [
            {"params": raw_model.encoder.parameters(), "lr": args.lr_backbone},
            {"params": list(raw_model.head_type.parameters()) + list(raw_model.head_make.parameters()) + list(raw_model.head_model.parameters()), "lr": args.lr_head},
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    w_type, w_make, w_model = args.loss_weights

    # Model EMA
    model_ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None
    if is_main_process and model_ema is not None:
        print(f"[*] Model EMA enabled with decay factor: {args.ema_decay}")

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Loss weights (Type, Make, Model): {args.loss_weights} | Label smoothing: {args.label_smoothing}")

    # Dry-run fast verification
    if args.dry_run:
        print("\n" + "=" * 70)
        print("RUNNING FINETUNE DRY-RUN VERIFICATION (2 batches of 4)")
        print("=" * 70)
        from torch.utils.data import Subset
        dry_train = DataLoader(Subset(train_dataset, range(min(8, len(train_dataset)))), batch_size=4, shuffle=True)
        model.train()
        for step, (images, meta) in enumerate(dry_train):
            if step >= 2:
                break
            images = images.to(device)
            y_t = meta["target_type"].to(device)
            y_m = meta["target_make"].to(device)
            y_mo = meta["target_model"].to(device)
            optimizer.zero_grad()
            out_t, out_m, out_mo = model(images)
            loss = w_type * criterion(out_t, y_t) + w_make * criterion(out_m, y_m) + w_model * criterion(out_mo, y_mo)
            loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            print(f"Dry-run step {step+1}/2 | Loss: {loss.item():.4f}")
        eval_mod = model_ema.module if model_ema is not None else model
        dry_val = DataLoader(Subset(val_dataset, range(min(4, len(val_dataset)))), batch_size=4, shuffle=False)
        m = evaluate_model(eval_mod, dry_val, taxonomy, device)
        print(f"Dry-run val marginal acc: {m['marginal_acc']:.2f}% | Exact: {m['acc_exact_match']:.2f}%")
        print("[✓] Dry-run complete. Exiting.\n")
        sys.exit(0)

    best_val_exact = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, (images, meta) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            y_t = meta["target_type"].to(device, non_blocking=True)
            y_m = meta["target_make"].to(device, non_blocking=True)
            y_mo = meta["target_model"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out_t, out_m, out_mo = model(images)
                    loss = (
                        w_type * criterion(out_t, y_t)
                        + w_make * criterion(out_m, y_m)
                        + w_model * criterion(out_mo, y_mo)
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out_t, out_m, out_mo = model(images)
                loss = (
                    w_type * criterion(out_t, y_t)
                    + w_make * criterion(out_m, y_m)
                    + w_model * criterion(out_mo, y_mo)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            total_loss += loss.item()

        scheduler.step()

        # Validation
        if is_main_process and (epoch % 2 == 0 or epoch == args.epochs):
            elapsed = time.time() - start_time
            eval_target = model_ema.module if model_ema is not None else model
            val_metrics = evaluate_model(eval_target, val_loader, taxonomy, device)
            tag = " (EMA)" if model_ema is not None else ""

            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                f"Loss: {total_loss/len(train_loader):.4f} | "
                f"Val Marginal Acc{tag}: {val_metrics['marginal_acc']:.2f}% "
                f"(T: {val_metrics['acc_type']:.2f}%, M: {val_metrics['acc_make']:.2f}%, Mo: {val_metrics['acc_model']:.2f}%) | "
                f"Exact Tuple: {val_metrics['acc_exact_match']:.2f}% | "
                f"Invalid: {val_metrics['pct_invalid_total']:.2f}% ({elapsed:.1f}s)"
            )

            if val_metrics["acc_exact_match"] > best_val_exact:
                best_val_exact = val_metrics["acc_exact_match"]
                best_ckpt_path = output_dir / f"best_model_fold{args.split_fold}.pth"
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "metrics": val_metrics,
                    "args": vars(args),
                }
                if model_ema is not None:
                    save_dict["ema_state_dict"] = model_ema.module.state_dict()
                    save_dict["model_state_dict"] = model_ema.module.state_dict()
                torch.save(save_dict, best_ckpt_path)

    # 5. Final Test Split Evaluation & Auto-Logging
    if is_main_process:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SPLIT")
        print("=" * 80)
        best_ckpt = torch.load(output_dir / f"best_model_fold{args.split_fold}.pth", map_location=device, weights_only=False)
        raw_model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate_model(model, test_loader, taxonomy, device)
        report_text = format_marginal_report(test_metrics)
        print(report_text)

        exp_name = args.exp_name if args.exp_name else f"ft_{args.backbone}_fold{args.split_fold}"
        save_experiment_result(
            experiment_name=exp_name,
            config=args,
            metrics=test_metrics,
            report_text=report_text,
            output_root=args.results_dir,
        )

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
