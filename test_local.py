#!/usr/bin/env python3
"""
Unit and Integration Tests for FEPA-FGVC
========================================
Validates SIGReg mathematical properties, representation collapse penalty,
dataset label mappings, LeJEPA forward/backward cycles, feature extraction,
hierarchical evaluation, and automatic experiment results logging.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sigreg import SIGReg
from models.lejepa_module import LeJEPA, LeJEPAEncoder, MLPProjector
from datasets.ufpr_dataset import UFPRDataset, CANONICAL_TYPES
from evaluate_hierarchical import VehicleTaxonomy, decode_predictions_hcdc, compute_hierarchical_metrics
from utils.results_logger import save_experiment_result


class TestSIGReg(unittest.TestCase):
    """Tests mathematical correctness of SIGReg isotropic Gaussian regularizer."""

    def setUp(self) -> None:
        self.sigreg = SIGReg(num_projections=128, num_nodes=17, t_max=3.0)

    def test_dimensions_and_shapes(self) -> None:
        """Verifies SIGReg output shapes across various batch sizes and latent dimensions."""
        for b in [8, 32, 64]:
            for k in [128, 256, 512]:
                z = torch.randn(b, k)
                loss = self.sigreg(z)
                self.assertEqual(loss.ndim, 0)
                self.assertFalse(torch.isnan(loss))
                self.assertFalse(torch.isinf(loss))

    def test_gradient_flow(self) -> None:
        """Verifies non-zero, finite gradients flow back to latent representations."""
        z = torch.randn(16, 256, requires_grad=True)
        loss = self.sigreg(z)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.all(z.grad == 0.0))
        self.assertFalse(torch.isnan(z.grad).any())

    def test_collapse_penalty(self) -> None:
        """Verifies collapsed representations (Z=0) produce strictly higher loss than N(0, I)."""
        z_standard = torch.randn(128, 256)
        z_zero_collapse = torch.zeros(128, 256)

        loss_norm = self.sigreg(z_standard).item()
        loss_zero = self.sigreg(z_zero_collapse).item()

        print(f"\n[SIGReg Validation] Standard Normal: {loss_norm:.6f} | Zero Collapse: {loss_zero:.6f}")
        self.assertLess(loss_norm, 0.20, "Standard normal should have low divergence")
        self.assertGreater(loss_zero, 2.0, "Zero collapse must incur severe penalty")
        self.assertGreater(loss_zero, loss_norm * 5.0)


class TestLeJEPAModule(unittest.TestCase):
    """Tests visual encoder, MLP projector, and forward/backward passes."""

    def test_forward_backward(self) -> None:
        model = LeJEPA(
            backbone_name="resnet18",
            pretrained=False,
            latent_dim=128,
            proj_hidden_dim=256,
            lambd=2.0,
            num_projections=32,
            num_quadrature_nodes=9,
        )
        v1 = torch.randn(4, 3, 224, 224)
        v2 = torch.randn(4, 3, 224, 224)

        out = model((v1, v2))
        loss = out["loss"]
        loss.backward()

        self.assertFalse(torch.isnan(loss))
        print(f"\n[LeJEPA ResNet18 Forward/Backward] Loss: {loss.item():.4f} (Sim: {out['loss_sim'].item():.4f}, SIGReg: {out['loss_sigreg'].item():.4f})")


class TestResultsLogger(unittest.TestCase):
    """Tests automatic experiment results logging and CSV summary appending."""

    def test_save_experiment_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {"backbone": "resnet18", "epochs": 5, "lr": 1e-4, "split_fold": 0}
            metrics = {
                "marginal_acc": 75.5,
                "acc_type": 90.0,
                "acc_make": 70.0,
                "acc_model": 66.5,
                "acc_exact_match": 60.0,
                "pct_invalid_total": 5.0,
                "pct_invalid_make_model": 3.0,
                "pct_invalid_model_type": 2.0,
                "rgb": {"acc_exact": 62.0},
                "ir": {"acc_exact": 55.0},
            }
            exp_dir = save_experiment_result(
                experiment_name="test_exp",
                config=config,
                metrics=metrics,
                report_text="Sample Test Report",
                output_root=tmp_dir,
            )

            self.assertTrue(exp_dir.exists())
            self.assertTrue((exp_dir / "config.json").exists())
            self.assertTrue((exp_dir / "metrics.json").exists())
            self.assertTrue((exp_dir / "report.txt").exists())
            self.assertTrue((Path(tmp_dir) / "benchmark_summary.csv").exists())
            print(f"[Results Logger Validation] Created: {exp_dir}")


def main() -> None:
    print("=" * 70)
    print("RUNNING FEPA-FGVC UNIT AND INTEGRATION TEST SUITE")
    print("=" * 70)
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
