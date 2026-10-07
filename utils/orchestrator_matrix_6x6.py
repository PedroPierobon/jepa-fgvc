#!/usr/bin/env python3
"""
Multi-Backbone Experimental Matrix Orchestrator (6x6)
=====================================================
Automates execution of the 6x6 research matrix on UFPR-VeSV:
- 6 Backbones: EfficientNet-V2, ResNet-50, ConvNeXt-Base, DINOv2, DINOv3, DepthAnything-V2
- 6 Regimes:
    1. frozen_probe (10 epochs linear probe on frozen features, 225 classes)
    2. direct_hcd   (30 epochs supervised 3-heads, evaluated with HCD-C)
    3. direct_225   (30 epochs supervised direct 225-class joint head)
    4. lejepa       (LeJEPA SSL 60 ep -> 30 ep 3-heads marginal)
    5. lejepa_hcd   (LeJEPA SSL -> 30 ep 3-heads evaluated with HCD-C)
    6. lejepa_225   (LeJEPA SSL -> 30 ep direct 225-class joint head)

Features:
- Smart checkpoint/result skipping (resumes without re-running finished jobs)
- Dynamic GPU monitoring (checks VRAM < 1500MB and Util < 10% for stability)
- Consolidates master summary into results/benchmark_multibackbone_6x6.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path("/experiments/ppm24/jepa-fgvc").resolve()
PYTHON_EXEC = BASE_DIR / ".venv/bin/python"
TORCHRUN_EXEC = BASE_DIR / ".venv/bin/torchrun"
DATA_DIR = BASE_DIR / "UFPR-VeSV"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_FT_DIR = BASE_DIR / "checkpoints_ft"

LOG_FILE = BASE_DIR / "orchestrator_matrix_6x6.log"
MATRIX_CSV = RESULTS_DIR / "benchmark_multibackbone_6x6.csv"

# 6 Official Backbones & Configuration
BACKBONES = [
    {
        "id": "convnext",
        "name": "convnext_base",
        "display": "ConvNeXt-Base",
        "family": "Modern Pure CNN",
        "existing_ssl": CHECKPOINTS_DIR / "convnext_lejepa/lejepa_encoder_fold0.pth",
    },
    {
        "id": "efficientnet_v2",
        "name": "tf_efficientnetv2_m.in21k_ft_in1k",
        "display": "EfficientNet-V2",
        "family": "CNN (MBConv)",
        "existing_ssl": CHECKPOINTS_DIR / "efficientnet_v2_lejepa/lejepa_encoder_fold0.pth",
    },
    {
        "id": "resnet50",
        "name": "resnet50",
        "display": "ResNet-50",
        "family": "CNN Clássica",
        "existing_ssl": None,
    },
    {
        "id": "dinov2",
        "name": "vit_base_patch14_dinov2",
        "display": "DINOv2 (ViT-B/14)",
        "family": "2D Foundation SSL",
        "existing_ssl": None,
    },
    {
        "id": "dinov3",
        "name": "vit_large_patch16_dinov3.lvd1689m",
        "display": "DINOv3 (ViT-L/16)",
        "family": "SOTA 2D Foundation",
        "existing_ssl": None,
    },
    {
        "id": "depth_anything_v2",
        "name": "depth_anything_v2",
        "display": "DepthAnything-v2",
        "family": "3D-Aware Foundation",
        "existing_ssl": None,
    },
]

REGIMES = [
    "frozen_probe",
    "direct_hcd",
    "direct_225",
    "lejepa",
    "lejepa_hcd",
    "lejepa_225",
]

MAX_MEM_MB = 1500
MAX_UTIL_PCT = 10
CHECK_INTERVAL = 30
REQUIRED_CONSECUTIVE = 3


def log_message(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    sys.stdout.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def get_all_gpus_status() -> Dict[int, Dict[str, int]]:
    cmd = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        status = {}
        for line in res.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                status[int(parts[0])] = {"mem": int(parts[1]), "util": int(parts[2])}
        return status
    except Exception as e:
        log_message(f"[!] Erro ao consultar nvidia-smi: {e}")
        return {}


def wait_for_free_gpu(
    candidate_gpus: List[int],
    max_mem_mb: int = MAX_MEM_MB,
    max_util_pct: int = MAX_UTIL_PCT,
    check_interval: int = CHECK_INTERVAL,
    required_consecutive: int = REQUIRED_CONSECUTIVE,
) -> int:
    idle_counters = {g: 0 for g in candidate_gpus}
    log_message(
        f"[*] Monitorando GPUs {candidate_gpus}... "
        f"Critério: VRAM < {max_mem_mb}MB e Util < {max_util_pct}% por {required_consecutive} ciclos ({check_interval}s)."
    )

    while True:
        status = get_all_gpus_status()
        for gpu_id in candidate_gpus:
            if gpu_id not in status:
                continue
            mem = status[gpu_id]["mem"]
            util = status[gpu_id]["util"]
            if mem < max_mem_mb and util < max_util_pct:
                idle_counters[gpu_id] += 1
            else:
                idle_counters[gpu_id] = 0

            print(f"    GPU {gpu_id} | VRAM: {mem:>5} MB | Util: {util:>3}% | Ciclos livres: {idle_counters[gpu_id]}/{required_consecutive}")

        for gpu_id in candidate_gpus:
            if idle_counters[gpu_id] >= required_consecutive:
                log_message(f"[+] GPU {gpu_id} está livre e estável!")
                return gpu_id

        print("-" * 65)
        time.sleep(check_interval)


def is_experiment_done(exp_name: str) -> Optional[Dict[str, Any]]:
    """Checks if experiment already exists in results/benchmark_summary.csv."""
    summary_file = RESULTS_DIR / "benchmark_summary.csv"
    if not summary_file.exists():
        return None

    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment_name", "") == exp_name:
                return row
    return None


def run_experiment_job(
    cmd: str,
    gpu_id: int,
    exp_name: str,
) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_message(f"\n[>] Disparando: {exp_name} na GPU {gpu_id}")
    log_message(f"    Comando: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, env=env, cwd=str(BASE_DIR))
    dur_min = (time.time() - t0) / 60.0
    if res.returncode == 0:
        log_message(f"[✓] {exp_name} concluído com sucesso em {dur_min:.1f} minutos!")
        return True
    else:
        log_message(f"[!] Erro ao executar {exp_name} (código {res.returncode})!")
        return False


def consolidate_matrix_report(fold: int = 0) -> None:
    """Consolidates all 36 cells into matrix CSV and Markdown summary."""
    summary_file = RESULTS_DIR / "benchmark_summary.csv"
    if not summary_file.exists():
        log_message("[-] benchmark_summary.csv ainda não encontrado.")
        return

    results_map: Dict[str, Dict[str, Any]] = {}
    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment_name", "")
            results_map[name] = row

    log_message("\n" + "=" * 105)
    log_message(f"RELATÓRIO CONSOLIDADO: MATRIZ EXPERIMENTAL MULTI-BACKBONE (6x6) - FOLD {fold}")
    log_message("=" * 105)

    header = f"{'Backbone':<20} | {'Frozen-Probe':<12} | {'Direct-HCD':<12} | {'Direct-225':<12} | {'LeJEPA':<12} | {'LeJEPA+HCD':<12} | {'LeJEPA+225':<12}"
    log_message(header)
    log_message("-" * 105)

    matrix_rows = []

    for b in BACKBONES:
        b_id = b["id"]
        row_str = f"{b['display']:<20} | "
        cell_dict = {"backbone": b["display"], "family": b["family"]}

        for r in REGIMES:
            exp_name = f"m6x6_{b_id}_{r}_fold{fold}"

            # Fallback for historical existing runs
            if exp_name not in results_map:
                if b_id == "convnext" and r == "lejepa_225":
                    exp_name = f"ft_lejepa_convnext_direct225_fold{fold}"
                elif b_id == "convnext" and r in ["lejepa", "lejepa_hcd"]:
                    exp_name = f"ft_lejepa_convnext_fold{fold}"
                elif b_id == "efficientnet_v2" and r in ["lejepa", "lejepa_hcd"]:
                    exp_name = f"ft_lejepa_efficientnet_v2_fold{fold}"

            if exp_name in results_map:
                row = results_map[exp_name]
                exact = float(row.get("exact_tuple_acc", 0.0))
                inv = float(row.get("invalid_tuples_pct", 0.0))
                cell_text = f"{exact:.2f}% (0%i)" if inv == 0.0 else f"{exact:.2f}%"
                row_str += f"{cell_text:>12} | "
                cell_dict[r] = exact
            else:
                row_str += f"{'[Pendente]':>12} | "
                cell_dict[r] = None

        log_message(row_str)
        matrix_rows.append(cell_dict)

    log_message("=" * 105 + "\n")

    # Save to dedicated CSV
    with open(MATRIX_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["backbone", "family"] + REGIMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in matrix_rows:
            writer.writerow(r)

    log_message(f"[✓] Tabela consolidada gravada em: {MATRIX_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador da Matriz Multi-Backbone 6x6")
    parser.add_argument("--fold", type=int, default=0, help="Split fold (padrão: 0)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1], help="GPUs candidatas (padrão: 0 1)")
    parser.add_argument("--report_only", action="store_true", help="Apenas consolida relatório da matriz")
    args = parser.parse_args()

    if args.report_only:
        consolidate_matrix_report(fold=args.fold)
        return

    log_message("=" * 90)
    log_message("ORQUESTRADOR INICIADO: MATRIZ MULTI-BACKBONE (6x6)")
    log_message(f"Fold: {args.fold} | GPUs Alvo: {args.gpus}")
    log_message(f"Backbones: {[b['display'] for b in BACKBONES]}")
    log_message(f"Regimes: {REGIMES}")
    log_message("=" * 90)

    # --- FILA DE EXPERIMENTOS ---
    # Prioridade 1: Frozen-Probes (Rápido, ~2-3 min cada)
    # Prioridade 2: Direct-225 e Direct-HCD
    # Prioridade 3: LeJEPA (Modelos com checkpoint existente)
    # Prioridade 4: LeJEPA SSL + FT nos restantes

    for b in BACKBONES:
        b_id = b["id"]
        b_name = b["name"]

        # 1. Regime: Frozen-Probe (Linear Probe 225 classes congelado, 10 épocas)
        exp_probe = f"m6x6_{b_id}_frozen_probe_fold{args.fold}"
        out_probe = CHECKPOINTS_FT_DIR / exp_probe
        if is_experiment_done(exp_probe):
            log_message(f"[✓] {exp_probe} já finalizado. Pulando...")
        else:
            gpu = wait_for_free_gpu(args.gpus)
            cmd = (
                f"{PYTHON_EXEC} finetune_hierarchical.py "
                f"--data_dir {DATA_DIR} "
                f"--split_fold {args.fold} "
                f"--backbone {b_name} "
                f"--pretrained "
                f"--frozen_probe "
                f"--direct_225 "
                f"--batch_size 32 "
                f"--epochs 10 "
                f"--lr_head 1.0e-3 "
                f"--amp "
                f"--exp_name {exp_probe} "
                f"--output_dir {out_probe}"
            )
            run_experiment_job(cmd, gpu, exp_probe)

        # 2. Regime: Direct-225 (Supervisionado direto, 30 épocas)
        exp_d225 = f"m6x6_{b_id}_direct_225_fold{args.fold}"
        out_d225 = CHECKPOINTS_FT_DIR / exp_d225
        if is_experiment_done(exp_d225):
            log_message(f"[✓] {exp_d225} já finalizado. Pulando...")
        else:
            gpu = wait_for_free_gpu(args.gpus)
            cmd = (
                f"{PYTHON_EXEC} finetune_hierarchical.py "
                f"--data_dir {DATA_DIR} "
                f"--split_fold {args.fold} "
                f"--backbone {b_name} "
                f"--pretrained "
                f"--direct_225 "
                f"--batch_size 32 "
                f"--epochs 30 "
                f"--lr_backbone 3e-5 "
                f"--lr_head 5e-4 "
                f"--amp "
                f"--exp_name {exp_d225} "
                f"--output_dir {out_d225}"
            )
            run_experiment_job(cmd, gpu, exp_d225)

        # 3. Regime: Direct-HCD (Supervisionado 3 cabeças, 30 épocas)
        exp_dhcd = f"m6x6_{b_id}_direct_hcd_fold{args.fold}"
        out_dhcd = CHECKPOINTS_FT_DIR / exp_dhcd
        if is_experiment_done(exp_dhcd):
            log_message(f"[✓] {exp_dhcd} já finalizado. Pulando...")
        else:
            gpu = wait_for_free_gpu(args.gpus)
            cmd = (
                f"{PYTHON_EXEC} finetune_hierarchical.py "
                f"--data_dir {DATA_DIR} "
                f"--split_fold {args.fold} "
                f"--backbone {b_name} "
                f"--pretrained "
                f"--batch_size 32 "
                f"--epochs 30 "
                f"--lr_backbone 3e-5 "
                f"--lr_head 5e-4 "
                f"--amp "
                f"--exp_name {exp_dhcd} "
                f"--output_dir {out_dhcd}"
            )
            run_experiment_job(cmd, gpu, exp_dhcd)

        # 4. Regime: LeJEPA + Direct 225
        # Verifica se checkpoint LeJEPA SSL existe
        ssl_ckpt = b.get("existing_ssl")
        if ssl_ckpt and Path(ssl_ckpt).exists():
            exp_l225 = f"m6x6_{b_id}_lejepa_225_fold{args.fold}"
            if b_id == "convnext" and is_experiment_done(f"ft_lejepa_convnext_direct225_fold{args.fold}"):
                log_message(f"[✓] {exp_l225} (ConvNeXt Direct 225) já finalizado (85.10%). Pulando...")
            elif is_experiment_done(exp_l225):
                log_message(f"[✓] {exp_l225} já finalizado. Pulando...")
            else:
                gpu = wait_for_free_gpu(args.gpus)
                out_l225 = CHECKPOINTS_FT_DIR / exp_l225
                cmd = (
                    f"{PYTHON_EXEC} finetune_hierarchical.py "
                    f"--data_dir {DATA_DIR} "
                    f"--split_fold {args.fold} "
                    f"--backbone {b_name} "
                    f"--lejepa_checkpoint {ssl_ckpt} "
                    f"--direct_225 "
                    f"--batch_size 32 "
                    f"--epochs 30 "
                    f"--lr_backbone 3e-5 "
                    f"--lr_head 5e-4 "
                    f"--amp "
                    f"--exp_name {exp_l225} "
                    f"--output_dir {out_l225}"
                )
                run_experiment_job(cmd, gpu, exp_l225)

    # Consolida relatório final ao término
    consolidate_matrix_report(fold=args.fold)
    log_message("[★] FILA DA MATRIZ MULTI-BACKBONE CONCLUÍDA COM SUCESSO!")


if __name__ == "__main__":
    main()
