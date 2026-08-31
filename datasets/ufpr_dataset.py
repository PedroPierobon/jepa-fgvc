"""
UFPR-VeSV Dataset Loader & Multi-View Augmentations
===================================================
PyTorch Dataset implementation for the UFPR-VeSV fine-grained vehicle surveillance dataset,
providing deterministic hierarchical label mappings (14 vehicle types, 26 makes, 136 models),
domain metadata (Visible Light / Daylight vs. Infrared / IR), and multi-view self-supervised
transformations.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Canonical 14 vehicle body types
CANONICAL_TYPES = [
    "SUV", "bus", "car", "compact-pickup", "compact-truck", "compact-van",
    "minibus", "motorcycle", "pickup", "scooter", "semi-trailer",
    "tractor-truck", "truck", "van"
]


class GaussianBlurTransform:
    """Applies random Gaussian blur with variable sigma."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class SolarizeTransform:
    """Applies random solarization by inverting pixels above threshold."""

    def __init__(self, threshold: int = 128) -> None:
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.solarize(img, threshold=self.threshold)


class LeJEPADataTransform:
    """
    Multi-view self-supervised data augmentation for LeJEPA.
    Generates two augmented global views designed to encourage domain invariance across
    Daylight (RGB) and Infrared (IR) surveillance imagery.
    """

    def __init__(
        self,
        img_size: int = 224,
        scale: Tuple[float, float] = (0.4, 1.0),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

        # View 1
        self.transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.25),  # Essential for cross-modal RGB/IR invariance
            transforms.RandomApply([GaussianBlurTransform(0.1, 2.0)], p=0.5),
            transforms.ToTensor(),
            self.normalize,
        ])

        # View 2
        self.transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.25),
            transforms.RandomApply([GaussianBlurTransform(0.1, 2.0)], p=0.1),
            transforms.RandomApply([SolarizeTransform(128)], p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform_1(img), self.transform_2(img)


def get_eval_transform(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Standard deterministic evaluation transform.
    """
    resize_dim = int(img_size * 256.0 / 224.0)
    return transforms.Compose([
        transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class UFPRDataset(Dataset):
    """
    PyTorch Dataset for UFPR-VeSV.

    Args:
        root_dir: Root directory of UFPR-VeSV (containing images/, annotations.json, splits/).
        split_fold: Fold index (0 to 9).
        subset: Dataset subset ('train', 'val', 'test').
        transform: Image transform callable.
        is_pretrain: If True and transform is None, applies LeJEPADataTransform.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split_fold: int = 0,
        subset: str = "train",
        transform: Optional[Callable] = None,
        is_pretrain: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split_fold = int(split_fold)
        self.subset = subset.lower()
        self.is_pretrain = is_pretrain

        if self.subset not in ["train", "val", "test"]:
            raise ValueError(f"subset must be 'train', 'val' or 'test', got '{subset}'")

        if not (0 <= self.split_fold <= 9):
            raise ValueError(f"split_fold must be between 0 and 9, got {split_fold}")

        self.images_dir = self.root_dir / "images"
        self.annotations_file = self.root_dir / "annotations.json"
        self.split_file = self.root_dir / "splits" / str(self.split_fold) / f"{self.subset}.txt"

        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        # 1. Load dataset annotations
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)

        # 2. Build deterministic alphabetical class mappings
        self._build_class_mappings(all_annotations)

        # 3. Map filename -> annotation dict
        self.annotations_map = {item["filename"]: item for item in all_annotations}

        # 4. Load split filenames
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        # 5. Define transforms
        if transform is not None:
            self.transform = transform
        else:
            if self.is_pretrain and self.subset == "train":
                self.transform = LeJEPADataTransform(img_size=224)
            else:
                self.transform = get_eval_transform(img_size=224)

    def _build_class_mappings(self, annotations: List[Dict[str, Any]]) -> None:
        """Builds alphabetically sorted index mappings for type, make, and model."""
        unique_types = sorted(list(set(item["type"] for item in annotations)))
        unique_makes = sorted(list(set(item["make"] for item in annotations)))
        unique_models = sorted(list(set(item["model"] for item in annotations)))

        self.type_to_idx = {name: idx for idx, name in enumerate(unique_types)}
        self.make_to_idx = {name: idx for idx, name in enumerate(unique_makes)}
        self.model_to_idx = {name: idx for idx, name in enumerate(unique_models)}

        self.idx_to_type = {idx: name for name, idx in self.type_to_idx.items()}
        self.idx_to_make = {idx: name for name, idx in self.make_to_idx.items()}
        self.idx_to_model = {idx: name for name, idx in self.model_to_idx.items()}

        self.num_types = len(self.type_to_idx)
        self.num_makes = len(self.make_to_idx)
        self.num_models = len(self.model_to_idx)

    def get_class_mappings(self) -> Dict[str, Any]:
        """Returns full mapping dictionaries."""
        return {
            "type_to_idx": self.type_to_idx,
            "make_to_idx": self.make_to_idx,
            "model_to_idx": self.model_to_idx,
            "idx_to_type": self.idx_to_type,
            "idx_to_make": self.idx_to_make,
            "idx_to_model": self.idx_to_model,
            "num_types": self.num_types,
            "num_makes": self.num_makes,
            "num_models": self.num_models,
        }

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Dict[str, Any]]:
        filename = self.filenames[index]
        img_path = self.images_dir / filename

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        raw_meta = self.annotations_map.get(filename, {})

        type_name = raw_meta.get("type", "unknown")
        make_name = raw_meta.get("make", "unknown")
        model_name = raw_meta.get("model", "unknown")
        is_ir_str = str(raw_meta.get("infrared", "no")).lower()
        is_infrared = (is_ir_str == "yes")

        target_type = self.type_to_idx.get(type_name, -1)
        target_make = self.make_to_idx.get(make_name, -1)
        target_model = self.model_to_idx.get(model_name, -1)

        metadata_dict = {
            "filename": filename,
            "target_type": torch.tensor(target_type, dtype=torch.long),
            "target_make": torch.tensor(target_make, dtype=torch.long),
            "target_model": torch.tensor(target_model, dtype=torch.long),
            "is_infrared": torch.tensor(is_infrared, dtype=torch.bool),
            "type_name": type_name,
            "make_name": make_name,
            "model_name": model_name,
            "plate": raw_meta.get("plate", ""),
            "rear_view": raw_meta.get("rear_view", "no").lower() == "yes",
            "corners": raw_meta.get("corners", []),
        }

        views = self.transform(image)
        return views, metadata_dict
