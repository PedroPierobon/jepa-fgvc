"""
VL-JEPA (Vision-Language Joint-Embedding Predictive Architecture) with Prototypes
================================================================================
Implements multimodal prototype-guided self-supervised pre-training aligning visual
representations directly with frozen semantic text prototypes in Euclidean space,
regularized by Sketched Isotropic Gaussian Regularization (SIGReg).

Key Innovations:
- Zero Contrastive Denial: Eliminates InfoNCE negative pairs and batch-size dependencies
  by anchoring visual features to taxonomic text prototypes via MSE in Euclidean space.
- Cross-View & Cross-Spectral Invariance: Preserves LeJEPA Latent-Euclidean view invariance
  across Daylight (RGB) and Nighttime Infrared (IR) surveillance imagery.
- Topological Regularization: SIGReg prevents latent space collapse and enforces isotropic
  geometry across the representation manifold.
- Zero-Shot Taxonomic Decoding: By construction, zero-shot predictions are 100% compliant
  with the vehicle catalog (0.00% invalid tuples).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lejepa_module import LeJEPAEncoder, MLPPredictor, MLPProjector
from .sigreg import SIGReg


class PrototypeTextBank(nn.Module):
    """
    Constructs, caches, and indexes frozen semantic text prototypes for all valid
    (Type, Make, Model) vehicle tuples in UFPR-VeSV.
    """

    def __init__(
        self,
        annotations_file: Union[str, Path],
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        prompt_template: str = "A photo of a {type}, make {make}, model {model}.",
        cache_path: Optional[Union[str, Path]] = None,
        scale_by_sqrt_dim: bool = True,
    ) -> None:
        super().__init__()
        from evaluate_hierarchical import VehicleTaxonomy

        self.annotations_file = Path(annotations_file)
        self.text_model_name = text_model_name
        self.prompt_template = prompt_template
        self.scale_by_sqrt_dim = scale_by_sqrt_dim

        # 1. Build ground truth taxonomy
        self.taxonomy = VehicleTaxonomy(self.annotations_file)
        self.num_types = self.taxonomy.num_types
        self.num_makes = self.taxonomy.num_makes
        self.num_models = self.taxonomy.num_models

        # Deterministically sort valid tuples
        self.tuples = sorted(list(self.taxonomy.valid_tuples_idx))
        self.num_tuples = len(self.tuples)

        # Build 3D lookup table [num_types, num_makes, num_models] -> tuple_idx
        lookup = torch.full(
            (self.num_types, self.num_makes, self.num_models),
            -1,
            dtype=torch.long,
        )
        for idx, (t, m, mo) in enumerate(self.tuples):
            lookup[t, m, mo] = idx

        self.register_buffer("lookup_table", lookup)
        self.register_buffer(
            "valid_tuples",
            torch.tensor(self.tuples, dtype=torch.long),
        )

        # 2. Encode or load text prototype embeddings
        prototypes_tensor = self._load_or_encode_prototypes(cache_path)
        self.embed_dim = prototypes_tensor.shape[1]
        self.register_buffer("prototypes", prototypes_tensor)

    def _load_or_encode_prototypes(
        self, cache_path: Optional[Union[str, Path]]
    ) -> torch.Tensor:
        """Loads cached embeddings or encodes them via SentenceTransformer."""
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                data = torch.load(cache_file, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and "prototypes" in data:
                    return data["prototypes"].float()
                elif isinstance(data, torch.Tensor):
                    return data.float()

        # Generate text prompts
        prompts = [
            self.prompt_template.format(
                type=self.taxonomy.idx_to_type[t],
                make=self.taxonomy.idx_to_make[m],
                model=self.taxonomy.idx_to_model[mo],
            )
            for t, m, mo in self.tuples
        ]

        from sentence_transformers import SentenceTransformer

        text_encoder = SentenceTransformer(self.text_model_name)
        embeddings = text_encoder.encode(prompts, convert_to_tensor=True, show_progress_bar=False)
        embeddings = embeddings.float().cpu()

        # Normalize to unit sphere
        embeddings = F.normalize(embeddings, p=2.0, dim=-1)

        # Scale by sqrt(D) so expected variance per coordinate is ~1.0
        if self.scale_by_sqrt_dim:
            embeddings = embeddings * math.sqrt(embeddings.shape[-1])

        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prototypes": embeddings, "tuples": self.tuples}, cache_file)

        return embeddings

    def get_targets(
        self,
        targets_type: torch.Tensor,
        targets_make: torch.Tensor,
        targets_model: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps batch class indices to prototype vectors and tuple indices.

        Args:
            targets_type: LongTensor [B]
            targets_make: LongTensor [B]
            targets_model: LongTensor [B]

        Returns:
            Tuple of (prototype_targets [B, D_text], tuple_indices [B])
        """
        device = targets_type.device
        tuple_idx = self.lookup_table[targets_type, targets_make, targets_model]
        proto_targets = self.prototypes[tuple_idx].to(device)
        return proto_targets, tuple_idx

    def classify_zero_shot(
        self, z: torch.Tensor, metric: str = "euclidean"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts valid vehicle tuples by comparing visual latents against prototypes.

        Args:
            z: Visual representations [B, D_text]
            metric: 'euclidean' (L2 distance) or 'cosine' (cosine similarity)

        Returns:
            Tuple (pred_tuple_idx [B], pred_type [B], pred_make [B], pred_model [B])
        """
        proto = self.prototypes.to(z.device)  # [225, D]

        if metric == "euclidean":
            # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
            # Fast vectorized squared Euclidean distance
            z_sq = (z ** 2).sum(dim=-1, keepdim=True)       # [B, 1]
            p_sq = (proto ** 2).sum(dim=-1, keepdim=True).t() # [1, 225]
            dists = z_sq + p_sq - 2.0 * torch.matmul(z, proto.t())  # [B, 225]
            best_tuple_idx = torch.argmin(dists, dim=-1)     # [B]
        elif metric == "cosine":
            z_norm = F.normalize(z, p=2.0, dim=-1)
            p_norm = F.normalize(proto, p=2.0, dim=-1)
            sims = torch.matmul(z_norm, p_norm.t())         # [B, 225]
            best_tuple_idx = torch.argmax(sims, dim=-1)      # [B]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        predicted_tuples = self.valid_tuples[best_tuple_idx]  # [B, 3]
        return (
            best_tuple_idx,
            predicted_tuples[:, 0],
            predicted_tuples[:, 1],
            predicted_tuples[:, 2],
        )


class VLJEPA(nn.Module):
    """
    Vision-Language Joint-Embedding Predictive Architecture (VL-JEPA).

    Integrates:
    1. Visual Encoder f_theta (ConvNeXt, ViT, Swin, EfficientNet)
    2. MLP Projector g_phi mapping visual representations to text latent space
    3. Optional Predictor p_psi
    4. PrototypeTextBank maintaining frozen semantic representations of all 225 valid tuples
    5. SIGReg regularizer preventing latent collapse and maintaining isotropic spread
    """

    def __init__(
        self,
        annotations_file: Union[str, Path],
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proj_hidden_dim: int = 2048,
        use_predictor: bool = False,
        lambd_sigreg: float = 2.0,
        num_projections: int = 128,
        num_quadrature_nodes: int = 17,
        t_max: float = 3.0,
        cache_prototypes_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.lambd_sigreg = float(lambd_sigreg)
        self.use_predictor = use_predictor

        # 1. Text Prototype Bank
        self.prototype_bank = PrototypeTextBank(
            annotations_file=annotations_file,
            text_model_name=text_model_name,
            cache_path=cache_prototypes_path,
        )
        self.text_dim = self.prototype_bank.embed_dim

        # 2. Visual Encoder
        self.encoder = LeJEPAEncoder(backbone_name=backbone_name, pretrained=pretrained)
        self.embed_dim = self.encoder.embed_dim

        # 3. Latent Projector mapping visual features [D_vis] -> text latent space [D_text]
        self.projector = MLPProjector(
            in_dim=self.embed_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=self.text_dim,
            num_layers=3,
        )

        # 4. Optional Predictor head
        self.predictor = (
            MLPPredictor(dim=self.text_dim, hidden_dim=proj_hidden_dim // 2)
            if use_predictor
            else nn.Identity()
        )

        # 5. SIGReg Regularizer
        self.sigreg = SIGReg(
            num_projections=num_projections,
            num_nodes=num_quadrature_nodes,
            t_max=t_max,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts backbone visual features."""
        return self.encoder(x)

    def forward_projector(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts projected representations in text latent space."""
        return self.projector(self.encoder(x))

    def forward_train(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        targets_type: torch.Tensor,
        targets_make: torch.Tensor,
        targets_model: torch.Tensor,
        alpha_proto: float = 1.0,
        beta_sim: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes multimodal prototype-guided VL-JEPA training step.
        """
        # Visual features
        f1 = self.encoder(v1)
        f2 = self.encoder(v2)

        # Projected latents in text embedding space
        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target text prototypes [B, D_text]
        proto_targets, _ = self.prototype_bank.get_targets(
            targets_type, targets_make, targets_model
        )

        # 1. Prototype Euclidean Alignment Loss (MSE without contrastive denominator)
        diff_proto_1 = z1 - proto_targets
        diff_proto_2 = z2 - proto_targets
        loss_proto = 0.5 * (
            torch.mean(diff_proto_1 ** 2) +
            torch.mean(diff_proto_2 ** 2)
        )

        # 2. Cross-View Latent-Euclidean Invariance Loss
        diff_sim_12 = p1 - z2
        diff_sim_21 = p2 - z1
        loss_sim = 0.5 * (
            torch.mean(diff_sim_12 ** 2) +
            torch.mean(diff_sim_21 ** 2)
        )

        # 3. SIGReg Regularization on concatenated views [2B, D_text]
        z_cat = torch.cat([z1, z2], dim=0)
        loss_sigreg = self.sigreg(z_cat)

        # Total Multimodal Objective
        total_loss = (
            alpha_proto * loss_proto +
            beta_sim * loss_sim +
            self.lambd_sigreg * loss_sigreg
        )

        return {
            "loss": total_loss,
            "loss_proto": loss_proto,
            "loss_sim": loss_sim,
            "loss_sigreg": loss_sigreg,
            "z1": z1,
            "z2": z2,
            "p1": p1,
            "p2": p2,
        }

    def classify_zero_shot(
        self, x: torch.Tensor, metric: str = "euclidean"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero-shot inference directly from input image."""
        z = self.forward_projector(x)
        return self.prototype_bank.classify_zero_shot(z, metric=metric)
