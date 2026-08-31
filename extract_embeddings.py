#!/usr/bin/env python3
"""
Feature Extraction for Downstream Hierarchical Evaluation
=========================================================
Extracts frozen visual representations from trained LeJEPA encoders or
ImageNet-pretrained backbones across train, val, and test splits of UFPR-VeSV.
Saves structured .pt files containing embeddings, ground truth targets,
and domain metadata (Daylight vs. Infrared).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.ufpr_dataset import UFPRDataset, get_eval_transform
from models.lejepa_module import LeJEPAEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature Extraction from Frozen Encoders on UFPR-VeSV"
    )
    parser.add_argument("--data_dir", type=str, default="./UFPR-VeSV", help="Root directory of UFPR-VeSV dataset")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained encoder checkpoint (.pth)")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights directly")
    parser.add_argument("--split_fold", type=int, default=0, help="Fold to process (0 to 9)")
    parser.add_argument("--subsets", nargs="+", default=["train", "val", "test"], help="Subsets to extract")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./extracted_embeddings", help="Directory to save .pt files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples per subset for quick debugging")

    return parser.parse_args()


@torch.no_grad()
def extract_features_for_subset(
    encoder: nn.Module,
    dataset: UFPRDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_samples: int = -1,
) -> Dict[str, Any]:
    """Extracts embeddings, targets, and metadata from dataset loader."""
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings: List[torch.Tensor] = []
    all_targets_type: List[torch.Tensor] = []
    all_targets_make: List[torch.Tensor] = []
    all_targets_model: List[torch.Tensor] = []
    all_is_infrared: List[torch.Tensor] = []
    all_filenames: List[str] = []

    samples_collected = 0
    for batch_idx, (images, meta) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        feats = encoder(images)  # [B, D]

        all_embeddings.append(feats.cpu())
        all_targets_type.append(meta["target_type"].cpu())
        all_targets_make.append(meta["target_make"].cpu())
        all_targets_model.append(meta["target_model"].cpu())
        all_is_infrared.append(meta["is_infrared"].cpu())
        all_filenames.extend(meta["filename"])

        samples_collected += images.size(0)
        if max_samples > 0 and samples_collected >= max_samples:
            break

    embeddings_cat = torch.cat(all_embeddings, dim=0)
    if max_samples > 0 and embeddings_cat.size(0) > max_samples:
        embeddings_cat = embeddings_cat[:max_samples]
        targets_type_cat = torch.cat(all_targets_type, dim=0)[:max_samples]
        targets_make_cat = torch.cat(all_targets_make, dim=0)[:max_samples]
        targets_model_cat = torch.cat(all_targets_model, dim=0)[:max_samples]
        is_ir_cat = torch.cat(all_is_infrared, dim=0)[:max_samples]
        all_filenames = all_filenames[:max_samples]
    else:
        targets_type_cat = torch.cat(all_targets_type, dim=0)
        targets_make_cat = torch.cat(all_targets_make, dim=0)
        targets_model_cat = torch.cat(all_targets_model, dim=0)
        is_ir_cat = torch.cat(all_is_infrared, dim=0)

    return {
        "embeddings": embeddings_cat,
        "targets_type": targets_type_cat,
        "targets_make": targets_make_cat,
        "targets_model": targets_model_cat,
        "is_infrared": is_ir_cat,
        "filenames": all_filenames,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print(f"FEATURE EXTRACTION - FOLD {args.split_fold}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else ('Pretrained ' + args.backbone if args.pretrained else 'None')}")
    print(f"Dataset root: {args.data_dir}")
    print(f"Subsets to process: {args.subsets}")
    print("=" * 70)

    # 1. Load encoder
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        backbone_name = ckpt.get("backbone", args.backbone) if isinstance(ckpt, dict) else args.backbone
        encoder = LeJEPAEncoder(backbone_name=backbone_name, pretrained=False)

        if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            encoder_keys = {
                k.replace("encoder.", "").replace("module.encoder.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if "encoder." in k
            }
            if encoder_keys:
                encoder.load_state_dict(encoder_keys)
            else:
                encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            encoder.load_state_dict(ckpt, strict=False)
        print(f"[✓] Loaded encoder weights from: {args.checkpoint}")
    else:
        encoder = LeJEPAEncoder(backbone_name=args.backbone, pretrained=args.pretrained)
        print(f"[✓] Initialized {args.backbone} (pretrained={args.pretrained})")

    encoder = encoder.to(device)
    encoder.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_transform = get_eval_transform(img_size=args.img_size)

    extracted_dict: Dict[str, Any] = {
        "metadata": {
            "backbone": args.backbone,
            "embed_dim": encoder.embed_dim,
            "split_fold": args.split_fold,
            "checkpoint": args.checkpoint,
            "pretrained": args.pretrained,
            "img_size": args.img_size,
        }
    }

    for subset in args.subsets:
        print(f"\n[*] Processing subset '{subset}'...")
        dataset = UFPRDataset(
            root_dir=args.data_dir,
            split_fold=args.split_fold,
            subset=subset,
            transform=eval_transform,
            is_pretrain=False,
        )

        subset_res = extract_features_for_subset(
            encoder=encoder,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            max_samples=args.max_samples,
        )
        extracted_dict[subset] = subset_res
        print(f"    Total samples: {len(subset_res['filenames']):,}")
        print(f"    [✓] Embeddings shape: {subset_res['embeddings'].shape}")

    consolidated_file = output_dir / f"embeddings_fold_{args.split_fold}.pt"
    torch.save(extracted_dict, consolidated_file)
    print(f"\n[✓] Consolidated embeddings saved to: {consolidated_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
