"""
LeJEPA Architecture (Latent-Euclidean Joint-Embedding Predictive Architecture)
=============================================================================
Implements visual encoder wrappers supporting timm and torchvision backbones,
non-linear projection heads, optional predictors, and normalized Latent-Euclidean
similarity objectives combined with SIGReg isotropic Gaussian regularization.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .sigreg import SIGReg


class MLPProjector(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Projector head mapping backbone features
    into regularized latent space.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        curr_dim = in_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim

        # Final linear layer without non-linearity
        layers.append(nn.Linear(curr_dim, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor(nn.Module):
    """
    Optional residual predictor head for asymmetric view prediction tasks.
    """

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 512,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=not use_bn),
            norm_layer,
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LeJEPAEncoder(nn.Module):
    """
    Unified visual encoder supporting modern backbones from timm and torchvision
    (e.g., EfficientNet-V2, Swin-V2, ConvNeXt, ViT-Base, ResNet-50).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        # 1. Try loading via timm
        if HAS_TIMM:
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,  # Remove classifier head for feature pooling
                    in_chans=in_chans,
                )
                self.embed_dim = self._get_backbone_dim()
                self.framework = "timm"
                return
            except Exception:
                pass

        # 2. Fallback to torchvision
        if HAS_TORCHVISION and hasattr(tv_models, backbone_name):
            weights = "DEFAULT" if pretrained else None
            model_fn = getattr(tv_models, backbone_name)
            try:
                self.backbone = model_fn(weights=weights)
            except TypeError:
                self.backbone = model_fn(pretrained=pretrained)

            if hasattr(self.backbone, "fc"):
                self.embed_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, "classifier"):
                if isinstance(self.backbone.classifier, nn.Linear):
                    self.embed_dim = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Identity()
                elif isinstance(self.backbone.classifier, nn.Sequential):
                    self.embed_dim = self.backbone.classifier[0].in_features
                    self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, "head"):
                self.embed_dim = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
            else:
                self.embed_dim = self._get_backbone_dim()

            self.framework = "torchvision"
            return

        raise ValueError(
            f"Backbone '{backbone_name}' could not be loaded via timm or torchvision."
        )

    def _get_backbone_dim(self) -> int:
        """Determines backbone output dimension using attributes or dummy forward."""
        if hasattr(self.backbone, "num_features") and self.backbone.num_features:
            return self.backbone.num_features
        if hasattr(self.backbone, "embed_dim") and self.backbone.embed_dim:
            return self.backbone.embed_dim
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            return self.backbone.head.in_features

        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy_input)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim > 2:
                feat = feat.mean(dim=[-2, -1])
            return feat.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat


class LeJEPA(nn.Module):
    """
    Full LeJEPA (Latent-Euclidean JEPA) pipeline with SIGReg regularization.

    Args:
        backbone_name: Encoder architecture (e.g. vit_base_patch16_224, swin_base_patch4_window7_224).
        pretrained: If True, initializes backbone from ImageNet weights.
        latent_dim: Dimension K of the regularized latent space.
        proj_hidden_dim: Hidden dimension of the MLP projector.
        lambd: Weight of SIGReg loss (L_total = L_sim + lambd * L_sigreg).
        num_projections: Number M of random directions in S^{K-1}.
        num_quadrature_nodes: Number P of symmetric quadrature nodes.
        t_max: Upper frequency limit for ECF integration.
        use_predictor: If True, uses residual predictor head for similarity loss.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        latent_dim: int = 256,
        proj_hidden_dim: int = 2048,
        lambd: float = 2.0,
        num_projections: int = 128,
        num_quadrature_nodes: int = 17,
        t_max: float = 3.0,
        use_predictor: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.lambd = float(lambd)
        self.use_predictor = use_predictor

        # 1. Visual Encoder f_theta
        self.encoder = LeJEPAEncoder(backbone_name=backbone_name, pretrained=pretrained)
        self.embed_dim = self.encoder.embed_dim

        # 2. Latent Projector g_phi: D -> K
        self.projector = MLPProjector(
            in_dim=self.embed_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=latent_dim,
            num_layers=3,
        )

        # 3. Optional Predictor p_psi: K -> K
        self.predictor = MLPPredictor(dim=latent_dim, hidden_dim=proj_hidden_dim // 2) if use_predictor else nn.Identity()

        # 4. SIGReg Regularizer
        self.sigreg = SIGReg(
            num_projections=num_projections,
            num_nodes=num_quadrature_nodes,
            t_max=t_max,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts frozen backbone representations (for downstream classification)."""
        return self.encoder(x)

    def forward_projector(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts regularized latent projection."""
        return self.projector(self.forward_features(x))

    def forward_views(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes training step for two augmented views of the same batch.
        """
        f1 = self.encoder(view1)
        f2 = self.encoder(view2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # 1. Normalized Mean Squared Error Latent Invariance Loss
        diff_12 = p1 - z2
        diff_21 = p2 - z1
        loss_sim = 0.5 * (
            torch.mean(diff_12 ** 2) +
            torch.mean(diff_21 ** 2)
        )

        # 2. SIGReg Loss evaluated on concatenated batch [2B, K]
        z_combined = torch.cat([z1, z2], dim=0)
        loss_sigreg = self.sigreg(z_combined)

        # 3. Total Loss: L = L_sim + lambd * L_SIGReg
        total_loss = loss_sim + self.lambd * loss_sigreg

        return {
            "loss": total_loss,
            "loss_sim": loss_sim,
            "loss_sigreg": loss_sigreg,
            "z1": z1,
            "z2": z2,
            "p1": p1,
            "p2": p2,
        }

    def forward(
        self,
        views: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(views, (tuple, list)):
            if len(views) >= 2:
                return self.forward_views(views[0], views[1])
            elif len(views) == 1:
                feats = self.forward_features(views[0])
                z = self.projector(feats)
                loss_sigreg = self.sigreg(z)
                return {"loss": loss_sigreg, "loss_sigreg": loss_sigreg, "z": z, "features": feats}
        elif isinstance(views, torch.Tensor):
            feats = self.forward_features(views)
            z = self.projector(feats)
            loss_sigreg = self.sigreg(z)
            return {"loss": loss_sigreg, "loss_sigreg": loss_sigreg, "z": z, "features": feats}

        raise ValueError("Unsupported input format for forward()")
