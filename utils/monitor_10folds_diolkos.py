#!/usr/bin/env python3
"""
Monitor e Orquestrador dos 10 Folds para Diolkos (FEPA-FGVC)
===========================================================
Monitora continuamente as GPUs da máquina diolkos (GPUs 0 e 1).
Quando uma GPU atinge estabilidade de ociosidade, dispara a fila de fine-tuning
dos 10 folds (0 a 9) para os backbones pré-treinados com LeJEPA.

Recursos:
1. Retomada Automática (Skip Existing): Pula automaticamente folds já concluídos.
2. Suporte a Múltiplos Backbones: EfficientNet-V2, ConvNeXt-Base, Swin-Base.
3. Avaliação Automática: Ao término de cada fold, executa a avaliação marginal e HCD-C.
4. Consolidação Final: Calcula Média e Desvio-Padrão (Mean ± Std) sobre os 10 folds,
   gerando relatórios prontos para o artigo oficial.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Configurações de Monitoramento da GPU ---
CANDIDATE_GPUS = [0, 1]          # GPUs candidatas
MAX_MEM_MB = 1000                # VRAM máxima usada para considerar livre (em MB)
MAX_UTIL_PCT = 10                # Utilização máxima tolerada (%)
CHECK_INTERVAL = 30              # Intervalo entre checagens (segundos)
REQUIRED_CONSECUTIVE = 3         # Ciclos seguidos ociosa para evitar falsos positivos

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXEC = BASE_DIR / ".venv/bin/python"
DATA_DIR = BASE_DIR / "UFPR-VeSV"
RESULTS_DIR = BASE_DIR / "results"

# --- Definição dos Backbones e seus Checkpoints LeJEPA ---
BACKBONE_CONFIGS = {
    "efficientnet_v2": {
        "name": "EfficientNet-V2 + LeJEPA",
        "backbone": "tf_efficientnetv2_m.in21k_ft_in1k",
        "alias": "efficientnet_v2",
        "lejepa_ckpt": str(BASE_DIR / "checkpoints/efficientnet_v2_lejepa/lejepa_encoder_fold0.pth"),
        "output_dir": str(BASE_DIR / "checkpoints_ft/efficientnet_v2_lejepa_ft"),
        "batch_size": 32,
        "lr_backbone": 3e-5,
        "lr_head": 5e-4,
        "epochs": 30,
    },
    "convnext": {
        "name": "ConvNeXt-Base + LeJEPA",
        "backbone": "convnext_base",
        "alias": "convnext",
        "lejepa_ckpt": str(BASE_DIR / "checkpoints/convnext_lejepa/lejepa_encoder_fold0.pth"),
        "output_dir": str(BASE_DIR / "checkpoints_ft/convnext_lejepa_ft"),
        "batch_size": 32,
        "lr_backbone": 3e-5,
        "lr_head": 5e-4,
        "epochs": 30,
    },
    "swin_base": {
        "name": "Swin-Base + LeJEPA",
        "backbone": "swin_base_patch4_window7_224",
        "alias": "swin_t",
        "lejepa_ckpt": str(BASE_DIR / "checkpoints/swin_t_lejepa/lejepa_encoder_fold0.pth"),
        "output_dir": str(BASE_DIR / "checkpoints_ft/swin_t_lejepa_ft"),
        "batch_size": 32,
        "lr_backbone": 3e-5,
        "lr_head": 5e-4,
        "epochs": 30,
    },
}


def get_all_gpus_status() -> Dict[int, Dict[str, int]]:
    """Consulta uso de VRAM e computação de todas as GPUs via nvidia-smi."""
    query = "index,memory.used,utilization.gpu"
    cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"
    gpu_data = {}
    try:
        output = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL).decode("utf-8").strip()
        for line in output.splitlines():
            if not line.strip():
                continue
            idx, mem, util = [int(x.strip()) for x in line.split(",")]
            gpu_data[idx] = {"mem": mem, "util": util}
        return gpu_data
    except Exception as e:
        print(f"[-] Erro ao executar nvidia-smi: {e}")
        return {}


def wait_for_free_gpu(candidate_gpus: List[int]) -> int:
    """Aguarda até que uma das GPUs candidatas fique ociosa por ciclos consecutivos."""
    idle_counters = {gpu_id: 0 for gpu_id in candidate_gpus}
    print(f"\n[*] Monitorando GPUs {candidate_gpus}...")
    print(f"[*] Critério: VRAM < {MAX_MEM_MB}MB e Util < {MAX_UTIL_PCT}% por {REQUIRED_CONSECUTIVE} ciclos ({CHECK_INTERVAL}s cada).")

    while True:
        status = get_all_gpus_status()
        timestamp = time.strftime("%H:%M:%S")

        for gpu_id in candidate_gpus:
            if gpu_id not in status:
                continue

            mem = status[gpu_id]["mem"]
            util = status[gpu_id]["util"]

            if mem < MAX_MEM_MB and util < MAX_UTIL_PCT:
                idle_counters[gpu_id] += 1
            else:
                idle_counters[gpu_id] = 0

            print(
                f"[{timestamp}] GPU {gpu_id} | VRAM: {mem:>5} MB | "
                f"Util: {util:>3}% | Ciclos livres: {idle_counters[gpu_id]}/{REQUIRED_CONSECUTIVE}"
            )

            if idle_counters[gpu_id] >= REQUIRED_CONSECUTIVE:
                print(f"\n[+] GPU {gpu_id} está livre e estável!")
                return gpu_id

        print("-" * 55)
        time.sleep(CHECK_INTERVAL)


def is_fold_completed(output_dir: str, fold: int) -> bool:
    """Verifica se o fold já possui checkpoint salvo."""
    ckpt_file = Path(output_dir) / f"best_model_fold{fold}.pth"
    return ckpt_file.exists() and ckpt_file.stat().st_size > 1000


def build_train_command(cfg: dict, fold: int) -> str:
    """Monta o comando de fine-tuning para um fold específico."""
    exp_name = f"ft_lejepa_{cfg['alias']}_fold{fold}"
    cmd = (
        f"{PYTHON_EXEC} finetune_hierarchical.py "
        f"--data_dir {DATA_DIR} "
        f"--split_fold {fold} "
        f"--backbone {cfg['backbone']} "
        f"--lejepa_checkpoint {cfg['lejepa_ckpt']} "
        f"--batch_size {cfg['batch_size']} "
        f"--lr_backbone {cfg['lr_backbone']} "
        f"--lr_head {cfg['lr_head']} "
        f"--epochs {cfg['epochs']} "
        f"--amp "
        f"--exp_name '{exp_name}' "
        f"--output_dir {cfg['output_dir']}"
    )
    return cmd


def consolidate_10folds(cfg_key: str):
    """Lê todos os 10 folds do master benchmark CSV e calcula Média ± Desvio-Padrão."""
    cfg = BACKBONE_CONFIGS[cfg_key]
    csv_file = RESULTS_DIR / "benchmark_summary.csv"
    if not csv_file.exists():
        print("[-] Arquivo benchmark_summary.csv não encontrado.")
        return

    pattern_alias = f"ft_lejepa_{cfg['alias']}_fold"
    folds_data = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row.get("experiment_name", "")
            if pattern_alias in exp_name:
                try:
                    # Extrai o fold
                    match = re.search(r"fold(\d+)", exp_name)
                    fold = int(match.group(1)) if match else int(row["split_fold"])
                    folds_data[fold] = {
                        "marginal_acc": float(row["marginal_acc"]),
                        "type_acc": float(row["type_acc"]),
                        "make_acc": float(row["make_acc"]),
                        "model_acc": float(row["model_acc"]),
                        "exact_tuple_acc": float(row["exact_tuple_acc"]),
                        "invalid_tuples_pct": float(row["invalid_tuples_pct"]),
                        "rgb_exact_acc": float(row["rgb_exact_acc"]),
                        "ir_exact_acc": float(row["ir_exact_acc"]),
                    }
                except (ValueError, KeyError):
                    continue

    print("\n" + "=" * 80)
    print(f"RELATÓRIO CONSOLIDADO DE 10 FOLDS: {cfg['name']}")
    print(f"Folds encontrados com sucesso: {len(folds_data)} / 10 ({sorted(list(folds_data.keys()))})")
    print("=" * 80)

    if not folds_data:
        print("[-] Nenhum dado registrado ainda.")
        return

    # Tabela por fold
    print(f"{'Fold':<6} | {'Marginal Mean':<14} | {'Exact Tuple':<12} | {'Invalid %':<10} | {'RGB Exact':<10} | {'IR Exact':<10}")
    print("-" * 80)
    for fold in sorted(folds_data.keys()):
        d = folds_data[fold]
        print(f"Fold {fold:<2} | {d['marginal_acc']:>12.2f}% | {d['exact_tuple_acc']:>10.2f}% | {d['invalid_tuples_pct']:>8.2f}% | {d['rgb_exact_acc']:>8.2f}% | {d['ir_exact_acc']:>8.2f}%")
    print("-" * 80)

    # Médias e desvios
    metrics_to_stat = ["marginal_acc", "type_acc", "make_acc", "model_acc", "exact_tuple_acc", "invalid_tuples_pct", "rgb_exact_acc", "ir_exact_acc"]
    summary_stats = {}
    for m in metrics_to_stat:
        values = [folds_data[f][m] for f in folds_data]
        summary_stats[m] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1 if len(values) > 1 else 0))}

    print("\n[★] RESULTADOS OFICIAIS (Mean ± Std):")
    print(f"  - Exact Tuple Accuracy:   {summary_stats['exact_tuple_acc']['mean']:.2f}% ± {summary_stats['exact_tuple_acc']['std']:.2f}%")
    print(f"  - Marginal Accuracy Mean: {summary_stats['marginal_acc']['mean']:.2f}% ± {summary_stats['marginal_acc']['std']:.2f}%")
    print(f"  - Type Accuracy (14):     {summary_stats['type_acc']['mean']:.2f}% ± {summary_stats['type_acc']['std']:.2f}%")
    print(f"  - Make Accuracy (26):     {summary_stats['make_acc']['mean']:.2f}% ± {summary_stats['make_acc']['std']:.2f}%")
    print(f"  - Model Accuracy (136):   {summary_stats['model_acc']['mean']:.2f}% ± {summary_stats['model_acc']['std']:.2f}%")
    print(f"  - Invalid Tuples Rate:    {summary_stats['invalid_tuples_pct']['mean']:.2f}% ± {summary_stats['invalid_tuples_pct']['std']:.2f}%")
    print(f"  - RGB Exact Accuracy:     {summary_stats['rgb_exact_acc']['mean']:.2f}% ± {summary_stats['rgb_exact_acc']['std']:.2f}%")
    print(f"  - IR (Infrared) Accuracy: {summary_stats['ir_exact_acc']['mean']:.2f}% ± {summary_stats['ir_exact_acc']['std']:.2f}%")
    print("=" * 80 + "\n")

    # Salvar JSON de resumo
    out_json = RESULTS_DIR / f"summary_10folds_{cfg_key}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"backbone": cfg["backbone"], "folds": folds_data, "stats": summary_stats}, f, indent=2)
    print(f"[✓] Resumo dos 10 folds salvo em: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Monitor e Orquestrador dos 10 Folds para Diolkos")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_v2",
        choices=list(BACKBONE_CONFIGS.keys()) + ["all"],
        help="Backbone a executar: efficientnet_v2, convnext, swin_base ou all",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)), help="Folds a executar (padrão: 0 1 2 3 4 5 6 7 8 9)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Pula folds já concluídos")
    parser.add_argument("--gpus", nargs="+", type=int, default=CANDIDATE_GPUS, help="GPUs candidatas para monitorar (padrão: 0 1)")
    parser.add_argument("--report_only", action="store_true", help="Apenas consolida os 10 folds sem rodar treinos")
    args = parser.parse_args()

    selected_backbones = list(BACKBONE_CONFIGS.keys()) if args.backbone == "all" else [args.backbone]

    if args.report_only:
        for b in selected_backbones:
            consolidate_10folds(b)
        return

    print("=" * 80)
    print("INICIANDO ORQUESTRADOR DOS 10 FOLDS (DIOLKOS)")
    print(f"Backbones selecionados: {selected_backbones}")
    print(f"Folds na fila: {args.folds}")
    print(f"GPUs candidatas: {args.gpus}")
    print("=" * 80)

    # Monta a fila de tarefas (backbone, fold)
    queue: List[Tuple[str, int]] = []
    for b in selected_backbones:
        cfg = BACKBONE_CONFIGS[b]
        for f in args.folds:
            if args.skip_existing and is_fold_completed(cfg["output_dir"], f):
                print(f"[*] Fold {f} do modelo {cfg['name']} já possui checkpoint concluído. Pulando...")
            else:
                queue.append((b, f))

    print(f"\n[+] Total de tarefas pendentes na fila: {len(queue)}")
    if not queue:
        print("[✓] Todas as tarefas já foram concluídas!")
        for b in selected_backbones:
            consolidate_10folds(b)
        return

    # Loop de consumo da fila com monitoramento
    while queue:
        b, fold = queue.pop(0)
        cfg = BACKBONE_CONFIGS[b]
        cmd = build_train_command(cfg, fold)

        print(f"\n" + "#" * 80)
        print(f"PRÓXIMA TAREFA NA FILA: {cfg['name']} | Fold {fold} ({len(queue)} restantes)")
        print(f"Comando: {cmd}")
        print("#" * 80)

        # Aguarda uma GPU livre
        free_gpu = wait_for_free_gpu(args.gpus)

        # Configura o ambiente com a GPU livre
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(free_gpu)

        print(f"\n[+] Alocando tarefa na GPU {free_gpu}...")
        start_time = time.time()
        process = subprocess.run(cmd, shell=True, env=env, cwd=str(BASE_DIR))
        elapsed_min = (time.time() - start_time) / 60.0

        if process.returncode == 0:
            print(f"[✓] Fold {fold} do {cfg['name']} finalizado com sucesso em {elapsed_min:.1f} minutos!")
        else:
            print(f"[-] Erro ao executar fold {fold} (código de retorno {process.returncode}). Recolocando no final da fila...")
            queue.append((b, fold))
            time.sleep(10)

    print("\n" + "=" * 80)
    print("TODAS AS TAREFAS DA FILA FORAM CONCLUÍDAS COM SUCESSO!")
    print("=" * 80)

    for b in selected_backbones:
        consolidate_10folds(b)


if __name__ == "__main__":
    main()
