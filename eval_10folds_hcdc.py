#!/usr/bin/env python3
"""
Avaliação Completa de 10 Folds com HCD e HCD-C (UFPR-VeSV)
==========================================================
Executa a decodificação Marginal, HCD Puro e HCD-C Calibrado
em todos os 10 folds (0 a 9) para os 3 modelos pré-treinados com LeJEPA:
1. EfficientNet-V2 + LeJEPA
2. ConvNeXt-Base + LeJEPA
3. Swin-Base + LeJEPA

Gera:
- Tabela individual por fold
- Estatísticas oficiais dos 10 folds: Média ± Desvio-Padrão (Mean ± Std)
- Ganhos absolutos (pp) e relativos (%)
- Desagregação em Luz Visível (RGB) e Noturno (Infravermelho - IR)
- Salvamento em CSV e JSON consolidado
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def evaluate_single_fold(
    name: str,
    backbone: str,
    ckpt_path: Path,
    fold: int,
    data_dir: Path,
    cat: Catalog,
    taxonomy: VehicleTaxonomy,
    device: torch.device,
    eval_transform,
) -> Dict[str, Any]:
    print(f"  -> Fold {fold}: Carregando checkpoint...", end=" ", flush=True)
    model = HierarchicalClassifier(backbone_name=backbone, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    val_dataset = UFPRDataset(root_dir=data_dir, split_fold=fold, subset="val", transform=eval_transform, is_pretrain=False)
    test_dataset = UFPRDataset(root_dir=data_dir, split_fold=fold, subset="test", transform=eval_transform, is_pretrain=False)

    print("Extraindo val/test...", end=" ", flush=True)
    z_val, y_val, _ = extract_logits_and_labels(model, val_dataset, batch_size=64, device=device)
    z_test, y_test, is_ir_test = extract_logits_and_labels(model, test_dataset, batch_size=64, device=device)

    # 1. Marginal
    marg_test = np.stack([zt.argmax(1) for zt in z_test], axis=1)
    acc_type_marg = (marg_test[:, 0] == y_test[:, 0]).mean() * 100.0
    acc_make_marg = (marg_test[:, 1] == y_test[:, 1]).mean() * 100.0
    acc_model_marg = (marg_test[:, 2] == y_test[:, 2]).mean() * 100.0
    marginal_mean_acc = (acc_type_marg + acc_make_marg + acc_model_marg) / 3.0

    exact_marg = (marg_test == y_test).all(axis=1)
    acc_exact_marg = exact_marg.mean() * 100.0
    hc_rate_marg = (~cat.contains(marg_test)).mean() * 100.0

    rgb_mask = ~is_ir_test
    ir_mask = is_ir_test
    acc_rgb_marg = exact_marg[rgb_mask].mean() * 100.0
    acc_ir_marg = exact_marg[ir_mask].mean() * 100.0

    # 2. HCD Puro
    hcd_pure = HCD(cat)
    pred_hcd_pure = hcd_pure.predict(z_test)
    exact_hcd_pure = (pred_hcd_pure == y_test).all(axis=1)
    acc_exact_hcd_pure = exact_hcd_pure.mean() * 100.0
    diag_pure = hcd_pure.diagnostics(z_test, y_test)
    acc_rgb_pure = exact_hcd_pure[rgb_mask].mean() * 100.0
    acc_ir_pure = exact_hcd_pure[ir_mask].mean() * 100.0

    # 3. HCD-C Calibrado
    hcdc = HCD(cat).fit(z_val, y_val, verbose=False)
    pred_hcdc = hcdc.predict(z_test)
    exact_hcdc = (pred_hcdc == y_test).all(axis=1)
    acc_exact_hcdc = exact_hcdc.mean() * 100.0
    diag_calib = hcdc.diagnostics(z_test, y_test)
    acc_rgb_calib = exact_hcdc[rgb_mask].mean() * 100.0
    acc_ir_calib = exact_hcdc[ir_mask].mean() * 100.0

    gain_abs_pure = acc_exact_hcd_pure - acc_exact_marg
    gain_abs_calib = acc_exact_hcdc - acc_exact_marg
    gain_rel_calib = (gain_abs_calib / acc_exact_marg) * 100.0

    print(f"Marg: {acc_exact_marg:.2f}% | HCD: {acc_exact_hcd_pure:.2f}% | HCD-C: {acc_exact_hcdc:.2f}% (Ganho: +{gain_abs_calib:.2f} pp)")

    return {
        "fold": fold,
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
            "invalid_pct": 0.0,
            "rescued": diag_pure["rescued"],
            "broken": diag_pure["broken"],
            "rgb_exact": acc_rgb_pure,
            "ir_exact": acc_ir_pure,
        },
        "hcd_calib": {
            "exact": acc_exact_hcdc,
            "gain_abs": gain_abs_calib,
            "gain_rel": gain_rel_calib,
            "invalid_pct": 0.0,
            "rescued": diag_calib["rescued"],
            "broken": diag_calib["broken"],
            "weights": hcdc.w.tolist(),
            "temperature": float(hcdc.T),
            "rgb_exact": acc_rgb_calib,
            "ir_exact": acc_ir_calib,
        },
    }


def compute_model_stats(folds_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(folds_results)
    stats = {}

    def get_mean_std(key_path):
        vals = []
        for r in folds_results:
            curr = r
            for k in key_path:
                curr = curr[k]
            vals.append(curr)
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1 if n > 1 else 0)),
            "values": [round(float(v), 2) for v in vals],
        }

    stats["marginal_exact"] = get_mean_std(["marginal", "exact"])
    stats["marginal_type"] = get_mean_std(["marginal", "type"])
    stats["marginal_make"] = get_mean_std(["marginal", "make"])
    stats["marginal_model"] = get_mean_std(["marginal", "model"])
    stats["marginal_mean"] = get_mean_std(["marginal", "mean"])
    stats["marginal_invalid"] = get_mean_std(["marginal", "invalid_pct"])
    stats["marginal_rgb"] = get_mean_std(["marginal", "rgb_exact"])
    stats["marginal_ir"] = get_mean_std(["marginal", "ir_exact"])

    stats["hcd_pure_exact"] = get_mean_std(["hcd_pure", "exact"])
    stats["hcd_pure_gain_abs"] = get_mean_std(["hcd_pure", "gain_abs"])
    stats["hcd_pure_rgb"] = get_mean_std(["hcd_pure", "rgb_exact"])
    stats["hcd_pure_ir"] = get_mean_std(["hcd_pure", "ir_exact"])

    stats["hcd_calib_exact"] = get_mean_std(["hcd_calib", "exact"])
    stats["hcd_calib_gain_abs"] = get_mean_std(["hcd_calib", "gain_abs"])
    stats["hcd_calib_gain_rel"] = get_mean_std(["hcd_calib", "gain_rel"])
    stats["hcd_calib_rgb"] = get_mean_std(["hcd_calib", "rgb_exact"])
    stats["hcd_calib_ir"] = get_mean_std(["hcd_calib", "ir_exact"])

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Avaliação de Folds com HCD e HCD-C")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Diretório de checkpoints a avaliar (contendo best_model_fold{i}.pth)")
    parser.add_argument("--backbone", type=str, default="convnext_base", help="Nome do backbone timm (default: convnext_base)")
    parser.add_argument("--model_name", type=str, default="ConvNeXt-Base + LeJEPA Recipe 50ep", help="Nome de exibição do modelo")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds específicos para avaliar (ex: --folds 0 4 9)")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR / "UFPR-VeSV"), help="Diretório do dataset UFPR-VeSV")
    parser.add_argument("--output_csv", type=str, default=None, help="Arquivo CSV de saída customizado")
    parser.add_argument("--output_json", type=str, default=None, help="Arquivo JSON de saída customizado")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de execução: {device}")

    data_dir = Path(args.data_dir)
    taxonomy = VehicleTaxonomy(data_dir / "annotations.json")
    cat = Catalog(
        taxonomy.valid_tuples_tensor.numpy(),
        task_names=["type", "make", "model"],
        n_classes=[14, 26, 136],
    )
    eval_transform = get_eval_transform(img_size=224)

    if args.ckpt_dir is not None:
        models = [
            (
                args.model_name,
                args.backbone,
                Path(args.ckpt_dir),
            )
        ]
    else:
        models = [
            (
                "EfficientNet-V2 + LeJEPA",
                "tf_efficientnetv2_m.in21k_ft_in1k",
                BASE_DIR / "checkpoints_ft/efficientnet_v2_lejepa_ft",
            ),
            (
                "ConvNeXt-Base + LeJEPA",
                "convnext_base",
                BASE_DIR / "checkpoints_ft/convnext_lejepa_ft",
            ),
            (
                "Swin-Base + LeJEPA",
                "swin_base_patch4_window7_224",
                BASE_DIR / "checkpoints_ft/swin_t_lejepa_ft",
            ),
        ]

    target_folds = args.folds if args.folds is not None else list(range(10))

    all_models_summary = {}
    master_csv_rows = []

    for name, backbone, ckpt_dir in models:
        print("\n" + "=" * 80)
        print(f"INICIANDO AVALIAÇÃO DE FOLDS {target_folds}: {name}")
        print("=" * 80)

        folds_results = []
        for fold in target_folds:
            ckpt_file = ckpt_dir / f"best_model_fold{fold}.pth"
            if not ckpt_file.exists():
                print(f"  [-] Aviso: Checkpoint {ckpt_file} não encontrado. Pulando...")
                continue

            res = evaluate_single_fold(
                name=name,
                backbone=backbone,
                ckpt_path=ckpt_file,
                fold=fold,
                data_dir=data_dir,
                cat=cat,
                taxonomy=taxonomy,
                device=device,
                eval_transform=eval_transform,
            )
            folds_results.append(res)

            # Gravar linha do CSV
            master_csv_rows.append({
                "model_name": name,
                "backbone": backbone,
                "fold": fold,
                "marginal_exact": f"{res['marginal']['exact']:.2f}",
                "marginal_mean": f"{res['marginal']['mean']:.2f}",
                "marginal_invalid": f"{res['marginal']['invalid_pct']:.2f}",
                "marginal_rgb": f"{res['marginal']['rgb_exact']:.2f}",
                "marginal_ir": f"{res['marginal']['ir_exact']:.2f}",
                "hcd_pure_exact": f"{res['hcd_pure']['exact']:.2f}",
                "hcd_pure_gain_abs": f"{res['hcd_pure']['gain_abs']:+.2f}",
                "hcd_pure_rgb": f"{res['hcd_pure']['rgb_exact']:.2f}",
                "hcd_pure_ir": f"{res['hcd_pure']['ir_exact']:.2f}",
                "hcdc_calib_exact": f"{res['hcd_calib']['exact']:.2f}",
                "hcdc_calib_gain_abs": f"{res['hcd_calib']['gain_abs']:+.2f}",
                "hcdc_calib_gain_rel": f"{res['hcd_calib']['gain_rel']:+.2f}",
                "hcdc_calib_rgb": f"{res['hcd_calib']['rgb_exact']:.2f}",
                "hcdc_calib_ir": f"{res['hcd_calib']['ir_exact']:.2f}",
                "hcdc_rescued": res["hcd_calib"]["rescued"],
                "hcdc_broken": res["hcd_calib"]["broken"],
            })

        stats = compute_model_stats(folds_results)
        all_models_summary[name] = {"backbone": backbone, "folds_results": folds_results, "stats": stats}

        print("\n" + "-" * 80)
        print(f"RESUMO DOS 10 FOLDS (Mean ± Std) - {name}:")
        print(f"  * Acurácia Marginal Exata:  {stats['marginal_exact']['mean']:.2f}% ± {stats['marginal_exact']['std']:.2f}% (Inválidas: {stats['marginal_invalid']['mean']:.2f}%)")
        print(f"  * Acurácia HCD Puro:        {stats['hcd_pure_exact']['mean']:.2f}% ± {stats['hcd_pure_exact']['std']:.2f}% (Ganho: +{stats['hcd_pure_gain_abs']['mean']:.2f} pp)")
        print(f"  * Acurácia HCD-C Calibrado: {stats['hcd_calib_exact']['mean']:.2f}% ± {stats['hcd_calib_exact']['std']:.2f}% (Ganho: +{stats['hcd_calib_gain_abs']['mean']:.2f} pp | Relativo: +{stats['hcd_calib_gain_rel']['mean']:.2f}%)")
        print(f"  * RGB Exact (HCD-C):        {stats['hcd_calib_rgb']['mean']:.2f}% ± {stats['hcd_calib_rgb']['std']:.2f}%")
        print(f"  * IR Exact (HCD-C):         {stats['hcd_calib_ir']['mean']:.2f}% ± {stats['hcd_calib_ir']['std']:.2f}%")
        print("-" * 80)

    # Salvar resultados consolidados
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_out = Path(args.output_json) if args.output_json else results_dir / "hcdc_10folds_all_models.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] JSON completo salvo em: {json_out}")

    csv_out = Path(args.output_csv) if args.output_csv else results_dir / "hcdc_10folds_summary.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "model_name", "backbone", "fold",
        "marginal_exact", "marginal_mean", "marginal_invalid", "marginal_rgb", "marginal_ir",
        "hcd_pure_exact", "hcd_pure_gain_abs", "hcd_pure_rgb", "hcd_pure_ir",
        "hcdc_calib_exact", "hcdc_calib_gain_abs", "hcdc_calib_gain_rel", "hcdc_calib_rgb", "hcdc_calib_ir",
        "hcdc_rescued", "hcdc_broken"
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(master_csv_rows)
    print(f"[✓] CSV master de 10 folds salvo em: {csv_out}")

    # Tabela Final Comparativa do Artigo Oficial
    print("\n" + "=" * 105)
    print("TABELA FINAL OFICIAL DOS 10 FOLDS (UFPR-VeSV): MARGINAL vs HCD vs HCD-C")
    print("=" * 105)
    print(f"{'Modelo':<26} | {'Marginal (Mean±Std)':<22} | {'HCD-C (Mean±Std)':<20} | {'Ganho Abs (pp)':<16} | {'Ganho Rel (%)':<14} | {'Tuplas Inválidas'}")
    print("-" * 105)
    for name, data in all_models_summary.items():
        st = data["stats"]
        m_str = f"{st['marginal_exact']['mean']:.2f} ± {st['marginal_exact']['std']:.2f}%"
        h_str = f"{st['hcd_calib_exact']['mean']:.2f} ± {st['hcd_calib_exact']['std']:.2f}%"
        g_abs = f"+{st['hcd_calib_gain_abs']['mean']:.2f} ± {st['hcd_calib_gain_abs']['std']:.2f} pp"
        g_rel = f"+{st['hcd_calib_gain_rel']['mean']:.2f}%"
        inv_str = f"{st['marginal_invalid']['mean']:.2f}% -> 0.00%"
        print(f"{name:<26} | {m_str:<22} | {h_str:<20} | {g_abs:<16} | {g_rel:<14} | {inv_str}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
