import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import json
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_label_from_dirname(dirname: str) -> str:
    """Normalize class label from a directory name like 'assault_violence (7)' -> 'assault_violence'."""
    base = dirname.strip()
    # Drop trailing parenthetical like " (7)"
    if base.endswith(')') and '(' in base:
        base = base[: base.rfind('(')].strip()
    return base


class FramesByClassDataset(Dataset):
    """Dataset that infers class labels from subdirectory names under a root folder of frames.

    Expected structure:
    data/extracted_frames/
      <class_or_video_name>/
        frame_000001.jpg
        ...

    This dataset normalizes directory names like 'assault_violence (7)' to class 'assault_violence'.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self._scan()

    def _scan(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Frames root not found: {self.root}")

        # Map normalized class -> list of image paths
        class_to_images: Dict[str, List[Path]] = {}
        for subdir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            label = _normalize_label_from_dirname(subdir.name)
            # Collect images in this subdir
            for img_path in subdir.glob("*.*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    class_to_images.setdefault(label, []).append(img_path)

        if not class_to_images:
            raise RuntimeError(f"No images found under {self.root}")

        classes = sorted(class_to_images.keys())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for cls, images in class_to_images.items():
            idx = self.class_to_idx[cls]
            for p in images:
                self.samples.append((p, idx))

        if len(self.class_to_idx) < 2:
            logger.warning(
                "Only one class detected. Training will proceed but metrics may be trivial."
            )

        logger.info(
            f"Discovered {len(self.samples)} images across {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


@dataclass
class TrainConfig:
    frames_root: str = "data/extracted_frames"
    output_dir: str = "ai_model"
    model_name: str = "mobilenet_v3_small"
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    val_split: float = 0.15
    num_workers: int = 2


class BehaviorCNNClassifier:
    """MobileNetV3-based behavior classifier trained on extracted frames."""

    def __init__(self, model_path: str = "ai_model/mobilenet_v3_behavior.pth"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.idx_to_class: Dict[int, str] = {}
        self.temperature: float = 1.0
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.model_path.exists():
            self._load()

    def _create_model(self, num_classes: int) -> nn.Module:
        if "large" in str(self.model_path).lower():
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        return backbone

    def train(self, config: TrainConfig) -> Dict:
        train_tf, val_tf = get_transforms(config.image_size)
        full_ds = FramesByClassDataset(config.frames_root, transform=None)

        # Save class mapping now
        class_to_idx = full_ds.class_to_idx
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        # Rebuild datasets with transforms
        full_ds.transform = train_tf
        # Split
        val_size = max(1, int(len(full_ds) * config.val_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        # Set val transform separately
        val_ds.dataset.transform = val_tf

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
        )

        # Initialize or resume model
        if self.model is None:
            self.model = self._create_model(num_classes=len(class_to_idx)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

        best_val_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [train]"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            # Validation
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [val]"):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_running_loss / max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            msg = (
                f"Epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )
            logger.info(msg)
            print(f"[CNN] {msg}")

            # Save checkpoint every epoch and keep best
            self._save(idx_to_class, config)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                self._save(idx_to_class, config)

        return {"best_val_acc": best_val_acc, "history": history, "classes": idx_to_class}

    def _save(self, idx_to_class: Dict[int, str], config: TrainConfig):
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "idx_to_class": idx_to_class,
            "model_name": config.model_name,
        }, out_dir / "mobilenet_v3_behavior.pth")
        with open(out_dir / "mobilenet_v3_behavior.meta.json", "w") as f:
            json.dump({
                "idx_to_class": idx_to_class,
                "image_size": config.image_size,
                "model_name": config.model_name,
                "temperature": getattr(self, 'temperature', 1.0),
            }, f, indent=2)
        logger.info(f"Saved best model and metadata to {out_dir}")

    def _load(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.idx_to_class = ckpt.get("idx_to_class", {})
        num_classes = len(self.idx_to_class) if self.idx_to_class else 2
        # Recreate model
        if "large" in self.model_path.name.lower():
            backbone = models.mobilenet_v3_large(weights=None)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            backbone = models.mobilenet_v3_small(weights=None)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = backbone.to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        # Try load temperature from meta
        try:
            meta_path = self.model_path.parent / "mobilenet_v3_behavior.meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.temperature = float(meta.get("temperature", 1.0))
        except Exception:
            self.temperature = 1.0
        logger.info(f"Loaded CNN model from {self.model_path} (T={self.temperature:.3f})")

    @torch.inference_mode()
    def predict_ndarray(self, frame_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None:
            raise RuntimeError("CNN model not loaded")
        # Convert BGR to RGB
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if self.temperature and self.temperature > 0:
            logits = logits / self.temperature
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        label = self.idx_to_class.get(idx.item(), str(idx.item()))
        # Build probs dict
        probs_dict: Dict[str, float] = {}
        for i in range(probs.size(0)):
            cls = self.idx_to_class.get(i, str(i))
            probs_dict[cls] = float(probs[i].item())
        return label, float(conf.item()), probs_dict

    def calibrate_temperature(self, val_loader: DataLoader, max_iters: int = 150, lr: float = 0.01) -> float:
        """Optimize temperature on validation set to minimize negative log-likelihood."""
        if self.model is None:
            self._load()
        self.model.eval()
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                logits_list.append(logits.detach())
                labels_list.append(labels.to(self.device))
        if not logits_list:
            raise RuntimeError("No validation data available for calibration")
        logits_cat = torch.cat(logits_list, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)

        log_T = torch.nn.Parameter(torch.tensor(0.0, device=self.device))  # T=1 initially
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iters)
        criterion = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = criterion(logits_cat / T, labels_cat)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_T).item())
        logger.info(f"Calibrated temperature T={self.temperature:.3f}")
        return self.temperature

    def calibrate_from_frames(self, frames_root: str, val_split: float = 0.15, batch_size: int = 64, image_size: int = 224) -> float:
        """Build a validation split from extracted frames and calibrate temperature automatically."""
        if self.model is None:
            self._load()
        _, val_tf = get_transforms(image_size)
        full_ds = FramesByClassDataset(frames_root, transform=val_tf)
        val_size = max(1, int(len(full_ds) * val_split))
        train_size = len(full_ds) - val_size
        if train_size <= 0:
            raise RuntimeError("Not enough data to split for calibration")
        _, val_ds = random_split(full_ds, [train_size, val_size])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        T = self.calibrate_temperature(val_loader)
        # Save updated temperature to meta
        meta_path = self.model_path.parent / "mobilenet_v3_behavior.meta.json"
        meta = {
            "idx_to_class": self.idx_to_class,
            "image_size": image_size,
            "model_name": "mobilenet_v3_small",
            "temperature": T,
        }
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save calibration meta: {e}")
        return T


