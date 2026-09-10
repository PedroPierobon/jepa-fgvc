#!/usr/bin/env python3
"""
Avaliação Comparativa de HCD e HCD-C nos Modelos LeJEPA Fine-Tunados
===================================================================
Aplica o HCD (puro) e HCD-C (calibrado no split de validação) nos backbones
pré-treinados com LeJEPA + SIGReg e fine-tunados no UFPR-VeSV Fold 0:
1. EfficientNet-V2 (tf_efficientnetv2_m.in21k_ft_in1k)
2. ConvNeXt-Base (convnext_base)
3. Swin-Base (swin_base_patch4_window7_224)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
REPO_HCD = BASE_DIR.parent / "HCD-C"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_HCD))

from datasets.ufpr_dataset import UFPRDataset, get_eval_transform
from finetune_hierarchical import HierarchicalClassifier
from evaluate_hierarchical import VehicleTaxonomy
from hcd import Catalog, HCD


@torch.no_grad()
def extract_logits_and_labels(model, dataset, batch_size=64, device="cuda"):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
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

    return z, y, is_ir


def evaluate_model(name, backbone, ckpt_path, val_dataset, test_dataset, cat, taxonomy, device):
    print(f"\n{'=' * 85}")
    print(f"AVALIANDO: {name} ({backbone})")
    print(f"{'=' * 85}")
    print(f"[*] Carregando checkpoint: {ckpt_path}")

    model = HierarchicalClassifier(backbone_name=backbone, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    print("[*] Extraindo logits de validação (para calibrar HCD-C)...")
    z_val, y_val, _ = extract_logits_and_labels(model, val_dataset, batch_size=64, device=device)

    print("[*] Extraindo logits de teste (5.075 amostras)...")
    z_test, y_test, is_ir_test = extract_logits_and_labels(model, test_dataset, batch_size=64, device=device)

    # 1. Decodificação Marginal (Baseline)
    marg_test = np.stack([zt.argmax(1) for zt in z_test], axis=1)
    acc_type_marg = (marg_test[:, 0] == y_test[:, 0]).mean() * 100.0
    acc_make_marg = (marg_test[:, 1] == y_test[:, 1]).mean() * 100.0
    acc_model_marg = (marg_test[:, 2] == y_test[:, 2]).mean() * 100.0
    marginal_mean_acc = (acc_type_marg + acc_make_marg + acc_model_marg) / 3.0

    exact_marg = (marg_test == y_test).all(axis=1)
    acc_exact_marg = exact_marg.mean() * 100.0
    hc_rate_marg = (~cat.contains(marg_test)).mean() * 100.0

    # RGB vs IR Marginal
    rgb_mask = ~is_ir_test
    ir_mask = is_ir_test
    acc_rgb_marg = exact_marg[rgb_mask].mean() * 100.0
    acc_ir_marg = exact_marg[ir_mask].mean() * 100.0

    # 2. HCD Puro (w=1, T=1)
    hcd_pure = HCD(cat)
    pred_hcd_pure = hcd_pure.predict(z_test)
    exact_hcd_pure = (pred_hcd_pure == y_test).all(axis=1)
    acc_exact_hcd_pure = exact_hcd_pure.mean() * 100.0
    diag_pure = hcd_pure.diagnostics(z_test, y_test)
    acc_rgb_pure = exact_hcd_pure[rgb_mask].mean() * 100.0
    acc_ir_pure = exact_hcd_pure[ir_mask].mean() * 100.0

    # 3. HCD-C Calibrado no Val
    print("[*] Calibrando pesos w e temperatura T no split de Validação...")
    hcdc = HCD(cat).fit(z_val, y_val, verbose=True)
    pred_hcdc = hcdc.predict(z_test)
    exact_hcdc = (pred_hcdc == y_test).all(axis=1)
    acc_exact_hcdc = exact_hcdc.mean() * 100.0
    diag_calib = hcdc.diagnostics(z_test, y_test)
    acc_rgb_calib = exact_hcdc[rgb_mask].mean() * 100.0
    acc_ir_calib = exact_hcdc[ir_mask].mean() * 100.0

    # Acurácias por tarefa com HCD-C
    acc_type_hcdc = (pred_hcdc[:, 0] == y_test[:, 0]).mean() * 100.0
    acc_make_hcdc = (pred_hcdc[:, 1] == y_test[:, 1]).mean() * 100.0
    acc_model_hcdc = (pred_hcdc[:, 2] == y_test[:, 2]).mean() * 100.0

    # Ganhos
    gain_abs_pure = acc_exact_hcd_pure - acc_exact_marg
    gain_rel_pure = (gain_abs_pure / acc_exact_marg) * 100.0

    gain_abs_calib = acc_exact_hcdc - acc_exact_marg
    gain_rel_calib = (gain_abs_calib / acc_exact_marg) * 100.0

    gain_over_pure = acc_exact_hcdc - acc_exact_hcd_pure

    print("\n--- RESULTADOS OBTIDOS ---")
    print(f"Decodificação Marginal (Baseline): {acc_exact_marg:.2f}% | Tuplas Inválidas: {hc_rate_marg:.2f}%")
    print(f"  -> RGB: {acc_rgb_marg:.2f}% | IR: {acc_ir_marg:.2f}%")
    print(f"HCD Puro (w=[1,1,1], T=1):         {acc_exact_hcd_pure:.2f}% (+{gain_abs_pure:.2f} pp | Relativo: +{gain_rel_pure:.2f}%)")
    print(f"  -> RGB: {acc_rgb_pure:.2f}% | IR: {acc_ir_pure:.2f}% | Resgatadas: {diag_pure['rescued']} | Broken: {diag_pure['broken']}")
    print(f"HCD-C Calibrado (Validação):       {acc_exact_hcdc:.2f}% (+{gain_abs_calib:.2f} pp | Relativo: +{gain_rel_calib:.2f}%)")
    print(f"  -> RGB: {acc_rgb_calib:.2f}% | IR: {acc_ir_calib:.2f}% | Resgatadas: {diag_calib['rescued']} | Broken: {diag_calib['broken']}")
    print(f"  -> Pesos aprendidos: w={hcdc.w.round(4).tolist()}, T={hcdc.T:.4f}")
    print(f"  -> Ganho adicional do HCD-C sobre HCD Puro: {gain_over_pure:+.2f} pp")

    return {
        "name": name,
        "backbone": backbone,
        "marginal": {
            "exact": acc_exact_marg,
            "type": acc_type_marg,
            "make": acc_make_marg,
            "model": acc_model_marg,
            "mean": marginal_mean_acc,
            "invalid_pct": hc_rate_marg,
            "rgb_exact": acc_rgb_marg,
            "ir_exact": acc_ir_marg,
        },
        "hcd_pure": {
            "exact": acc_exact_hcd_pure,
            "gain_abs": gain_abs_pure,
            "gain_rel": gain_rel_pure,
            "invalid_pct": 0.0,
            "rescued": diag_pure["rescued"],
            "broken": diag_pure["broken"],
            "rgb_exact": acc_rgb_pure,
            "ir_exact": acc_ir_pure,
        },
        "hcd_calib": {
            "exact": acc_exact_hcdc,
            "type": acc_type_hcdc,
            "make": acc_make_hcdc,
            "model": acc_model_hcdc,
            "gain_abs": gain_abs_calib,
            "gain_rel": gain_rel_calib,
            "gain_over_pure": gain_over_pure,
            "invalid_pct": 0.0,
            "rescued": diag_calib["rescued"],
            "broken": diag_calib["broken"],
            "weights": hcdc.w.tolist(),
            "temperature": hcdc.T,
            "rgb_exact": acc_rgb_calib,
            "ir_exact": acc_ir_calib,
        },
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de execução: {device}")

    data_dir = BASE_DIR / "UFPR-VeSV"
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")
    cat = Catalog(
        taxonomy.valid_tuples_tensor.numpy(),
        task_names=["type", "make", "model"],
        n_classes=[14, 26, 136],
    )
    print(f"Catálogo construído: {cat}")

    eval_transform = get_eval_transform(img_size=224)
    val_dataset = UFPRDataset(root_dir=data_dir, split_fold=0, subset="val", transform=eval_transform, is_pretrain=False)
    test_dataset = UFPRDataset(root_dir=data_dir, split_fold=0, subset="test", transform=eval_transform, is_pretrain=False)

    models_to_eval = [
        (
            "EfficientNet-V2 + LeJEPA",
            "tf_efficientnetv2_m.in21k_ft_in1k",
            BASE_DIR / "checkpoints_ft/efficientnet_v2_lejepa_ft/best_model_fold0.pth",
        ),
        (
            "ConvNeXt-Base + LeJEPA",
            "convnext_base",
            BASE_DIR / "checkpoints_ft/convnext_lejepa_ft/best_model_fold0.pth",
        ),
        (
            "Swin-Base + LeJEPA",
            "swin_base_patch4_window7_224",
            BASE_DIR / "checkpoints_ft/swin_t_lejepa_ft/best_model_fold0.pth",
        ),
    ]

    all_results = []
    for name, bb, ckpt in models_to_eval:
        res = evaluate_model(name, bb, ckpt, val_dataset, test_dataset, cat, taxonomy, device)
        all_results.append(res)

    print("\n" + "=" * 95)
    print("TABELA CONSOLIDADA: IMPACTO DO HCD E HCD-C NOS BACKBONES COM LEJEPA")
    print("=" * 95)
    print(f"{'Modelo':<26} | {'Marginal':<9} | {'HCD Puro':<11} | {'HCD-C Calib':<12} | {'Ganho pp':<9} | {'Ganho %':<9} | {'Tuplas Inválidas'}")
    print("-" * 95)
    for r in all_results:
        m = r["marginal"]
        hp = r["hcd_pure"]
        hc = r["hcd_calib"]
        print(
            f"{r['name']:<26} | {m['exact']:>8.2f}% | {hp['exact']:>10.2f}% | {hc['exact']:>11.2f}% | {hc['gain_abs']:>+8.2f}p | {hc['gain_rel']:>+8.2f}% | {m['invalid_pct']:>5.2f}% -> 0.00%"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
