# FEPA-FGVC: Latent-Euclidean JEPA with SIGReg for Fine-Grained Vehicle Categorization

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-0.9+-green.svg)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture)** regularized with **SIGReg (Sketched Isotropic Gaussian Regularization)** and **Hierarchical Calibrated Decoding (HCD-C)** for Fine-Grained Vehicle Categorization (FGVC) on the **UFPR-VeSV** surveillance dataset.

---

## 📌 Key Highlights

- **Self-Supervised Pre-Training (LeJEPA + SIGReg):** Differentiable non-contrastive representation learning preventing collapse via random 1D projections (Cramér-Wold device) and Epps-Pulley empirical characteristic function testing.
- **Hierarchical Taxonomic Structure:** Evaluates fine-grained vehicle recognition across 3 hierarchical levels:
  - **14 Vehicle Types** (*car, SUV, truck, bus, motorcycle, etc.*)
  - **26 Vehicle Makes** (*Volkswagen, Fiat, Chevrolet, Ford, Toyota, etc.*)
  - **136 Vehicle Models** (*Gol, Palio, Onix, Corolla, Civic, etc.*)
  - Valid taxonomic subspace: **225 real combinations** out of 49,504 combinatorial product space ($0.45\%$).
- **Taxonomic Consistency & Invalid Tuple Assessment:** Quantifies cross-level classification inconsistencies (Make $\leftrightarrow$ Model, Model $\leftrightarrow$ Type) and eliminates impossible vehicle predictions via Constrained Hierarchical Decoding (**HCD-C**).
- **Automated Experiment Tracking:** Automatically saves structured JSON configurations, metrics, human-readable reports, and appends summary rows into `results/benchmark_summary.csv`.
- **Multi-GPU & AMP Support:** Fully equipped with PyTorch Distributed Data Parallel (DDP via `torchrun`) and Automatic Mixed Precision (`torch.amp.autocast`).

---

## 📁 Repository Structure

```text
fepa-fgvc/
├── datasets/
│   ├── __init__.py
│   └── ufpr_dataset.py           # UFPR-VeSV dataset loader, label mappings & transforms
├── models/
│   ├── __init__.py
│   ├── sigreg.py                 # SIGReg Isotropic Gaussian Regularizer (Epps-Pulley + Quadrature)
│   └── lejepa_module.py          # LeJEPA encoder, MLP projectors, and loss formulation
├── utils/
│   ├── __init__.py
│   └── results_logger.py         # Automated results, configs, and CSV summary logger
├── train_lejepa.py               # Self-Supervised pre-training script (DDP + AMP)
├── extract_embeddings.py         # Feature extraction script (.pt embeddings)
├── evaluate_hierarchical.py      # Marginal evaluation & invalid tuple assessment
├── finetune_hierarchical.py      # End-to-end supervised fine-tuning & benchmarking
├── test_local.py                 # Automated unit and integration test suite
├── requirements.txt              # Environment dependencies
├── .gitignore                    # Git ignore rules (protects dataset & checkpoints)
└── README.md
```

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/fepa-fgvc.git
cd fepa-fgvc

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Setup (UFPR-VeSV)

Place the UFPR-VeSV dataset inside the project root:
```text
UFPR-VeSV/
├── images/             # 24,945 vehicle images (RGB and ~21.5% Infrared)
├── annotations.json    # Metadata containing type, make, model, plate, infrared flags
└── splits/             # 10 folds (0 to 9) with train.txt, val.txt, test.txt
```

---

## 🛠️ Usage Guide

### 1. Self-Supervised Pre-Training (LeJEPA + SIGReg)

Train the model in self-supervised mode on 2 GPUs using `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_lejepa.py \
    --data_dir ./UFPR-VeSV \
    --split_fold 0 \
    --backbone swin_base_patch4_window7_224 \
    --pretrained \
    --latent_dim 256 \
    --proj_hidden_dim 2048 \
    --lambd 2.0 \
    --batch_size 64 \
    --lr 2.5e-4 \
    --epochs 60 \
    --amp \
    --output_dir ./checkpoints/swin_lejepa_fold0
```

### 2. Feature Extraction

Extract frozen representations across `train`, `val`, and `test` splits:

```bash
python extract_embeddings.py \
    --data_dir ./UFPR-VeSV \
    --checkpoint ./checkpoints/swin_lejepa_fold0/lejepa_encoder_fold0.pth \
    --split_fold 0 \
    --subsets train val test \
    --batch_size 128 \
    --output_dir ./extracted_embeddings/swin_lejepa \
    --device cuda
```

### 3. Hierarchical Marginal Evaluation

Evaluate marginal classification accuracies, exact-match tuple accuracy, and invalid tuple rates:

```bash
python evaluate_hierarchical.py \
    --data_dir ./UFPR-VeSV \
    --embeddings_file ./extracted_embeddings/swin_lejepa/embeddings_fold_0.pt \
    --test_subset test \
    --epochs 50 \
    --lr 3e-3 \
    --device cuda
```
*(Results are automatically formatted and saved to `results/` and `results/benchmark_summary.csv`)*.

### 4. End-to-End Supervised Fine-Tuning

Compare supervised fine-tuning directly against ImageNet weights or initialized from LeJEPA:

```bash
# Fine-tuning initialized from LeJEPA pre-trained weights
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_hierarchical.py \
    --data_dir ./UFPR-VeSV \
    --split_fold 0 \
    --backbone tf_efficientnetv2_m.in21k_ft_in1k \
    --lejepa_checkpoint ./checkpoints/swin_lejepa_fold0/lejepa_encoder_fold0.pth \
    --batch_size 32 \
    --lr_backbone 3e-5 \
    --lr_head 5e-4 \
    --epochs 30 \
    --amp \
    --output_dir ./checkpoints_ft/efficientnet_v2_lejepa
```

---

## 📈 Benchmark Results (UFPR-VeSV Test Set)

| Paradigm / Model | Type (14) | Make (26) | Model (136) | **Marginal Acc** | **Exact Tuple (T+M+M)** | **Invalid Tuples %** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VLM: PE-Core-L/14-336 + LoRA** | **97.98%** | **96.96%** | **88.53%** | **94.49%** | **86.18%** | *N/A* |
| **EfficientNet-V2 (Supervised Benchmark)** | — | — | — | — | **86.30%** | *N/A* |
| **Swin-T (End-to-End Supervised)** | ~96.5% | ~88.2% | ~74.5% | **86.40%** | ~68.0% | **6.27%** |
| **LeJEPA Swin-T (SSL + Linear Probe)** | **84.49%** | **55.76%** | **48.49%** | **62.91%** | **34.40%** | **36.57%** |
| **LeJEPA Swin-T + HCD-C (Calibrated)** | **85.14%** | **60.51%** | **47.23%** | **64.29%** | **44.67%** | **0.00%** |
| **YOLO11n (Flat Multi-Head)** | ~68.0% | ~41.0% | ~23.0% | **44.00%** | ~18.0% | **33.00%** |

---

## 🧪 Unit Tests

Run the complete test suite to verify SIGReg properties, dataset mappings, and pipeline integrity:

```bash
python test_local.py
```

---

## 📜 Citation & License

This project is licensed under the MIT License.
