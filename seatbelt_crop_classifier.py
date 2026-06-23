#!/usr/bin/env python3
"""
Train/evaluate a binary seatbelt classifier from a YOLOv8 dataset.

The dataset labels are expected to contain:
  2: person-noseatbelt -> class 0
  3: person-seatbelt   -> class 1

Usage:
  python seatbelt_crop_classifier.py train --data "/path/to/seat belt.v1i.yolov8"
  python seatbelt_crop_classifier.py predict --weights runs/seatbelt_crop/best.pt --image crop.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PERSON_NO_SEATBELT = 2
PERSON_SEATBELT = 3
LABEL_MAP = {PERSON_NO_SEATBELT: 0, PERSON_SEATBELT: 1}
CLASS_NAMES = ["not_worn", "worn"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class CropSample:
    image_path: Path
    label: int
    box_xywhn: tuple[float, float, float, float]


def yolo_to_xyxy(
    box_xywhn: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float,
) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box_xywhn
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height

    px = (x2 - x1) * pad
    py = (y2 - y1) * pad
    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(width, int(round(x2 + px)))
    y2 = min(height, int(round(y2 + py)))
    return x1, y1, x2, y2


def matching_image(label_path: Path, images_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(data_root: Path, split: str) -> list[CropSample]:
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    samples: list[CropSample] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = matching_image(label_path, images_dir)
        if image_path is None:
            continue

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id not in LABEL_MAP:
                continue
            box = tuple(float(x) for x in parts[1:5])
            samples.append(CropSample(image_path, LABEL_MAP[cls_id], box))  # type: ignore[arg-type]

    return samples


def limit_balanced(samples: list[CropSample], limit: int | None, seed: int) -> list[CropSample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    half = max(1, limit // 2)
    picked: list[CropSample] = []
    for label in (0, 1):
        bucket = by_label[label]
        rng.shuffle(bucket)
        picked.extend(bucket[: min(half, len(bucket))])
    if len(picked) < limit:
        rest = [s for s in samples if s not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: limit - len(picked)])
    rng.shuffle(picked)
    return picked


class SeatbeltCropDataset(Dataset):
    def __init__(self, samples: list[CropSample], train: bool, crop_pad: float, image_size: int):
        self.samples = samples
        self.crop_pad = crop_pad
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.2, 0.05)], p=0.8),
                    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            box = yolo_to_xyxy(sample.box_xywhn, img.width, img.height, self.crop_pad)
            crop = img.crop(box)
        return self.tf(crop), torch.tensor(sample.label, dtype=torch.long)


def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def make_loader(
    samples: list[CropSample],
    train: bool,
    batch_size: int,
    image_size: int,
    crop_pad: float,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = SeatbeltCropDataset(samples, train=train, crop_pad=crop_pad, image_size=image_size)
    if not train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    labels = np.array([s.label for s in samples])
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.DoubleTensor([weights[s.label] for s in samples])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=pin_memory)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_prob.extend(probs.cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),
        "n": len(y_true),
        "positive_rate": float(np.mean(y_pred)) if y_pred else 0.0,
        "mean_worn_probability": float(np.mean(y_prob)) if y_prob else 0.0,
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    data_root = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = collect_samples(data_root, "train")
    valid_samples = collect_samples(data_root, "valid")
    test_samples = collect_samples(data_root, "test")
    train_samples = limit_balanced(train_samples, args.limit_samples, args.seed)
    valid_samples = limit_balanced(valid_samples, args.limit_valid, args.seed)
    test_samples = limit_balanced(test_samples, args.limit_valid, args.seed)

    print(f"samples: train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples)}", flush=True)
    print(
        "train labels:",
        {name: sum(s.label == i for s in train_samples) for i, name in enumerate(CLASS_NAMES)},
        flush=True,
    )
    print(
        "valid labels:",
        {name: sum(s.label == i for s in valid_samples) for i, name in enumerate(CLASS_NAMES)},
        flush=True,
    )
    print(
        "test labels:",
        {name: sum(s.label == i for s in test_samples) for i, name in enumerate(CLASS_NAMES)},
        flush=True,
    )

    if not train_samples or not valid_samples:
        raise SystemExit("Missing train/valid samples. Check the dataset path and YOLO label classes.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device, flush=True)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_samples, True, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory
    )
    valid_loader = make_loader(
        valid_samples, False, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory
    )
    test_loader = (
        make_loader(test_samples, False, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory)
        if test_samples
        else None
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()

        valid_metrics = evaluate(model, valid_loader, device)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"loss={running_loss / max(seen, 1):.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_pos_rate={valid_metrics['positive_rate']:.3f}",
            flush=True,
        )

        if valid_metrics["accuracy"] > best_acc:
            best_acc = valid_metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "crop_pad": args.crop_pad,
                    "valid_metrics": valid_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final = {"valid": evaluate(model, valid_loader, device)}
    if test_loader is not None:
        final["test"] = evaluate(model, test_loader, device)

    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    print("\nBEST WEIGHTS:", best_path, flush=True)
    print("\nVALID REPORT\n", final["valid"]["report"], flush=True)
    if "test" in final:
        print("\nTEST REPORT\n", final["test"]["report"], flush=True)
    print("metrics saved:", out_dir / "metrics.json", flush=True)


def predict(args: argparse.Namespace) -> None:
    weights = Path(args.weights).expanduser().resolve()
    checkpoint = torch.load(weights, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 224))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with Image.open(Path(args.image).expanduser()) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        prob_worn = torch.softmax(model(tensor), dim=1)[0, 1].item()
    label = "worn" if prob_worn >= args.threshold else "not_worn"
    print(json.dumps({"label": label, "prob_worn": prob_worn, "threshold": args.threshold}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--data", default="/kaggle/input/datasets/manyaj123456/setabelt1")
    train_p.add_argument("--out", default="runs/seatbelt_crop")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=48)
    train_p.add_argument("--image-size", type=int, default=224)
    train_p.add_argument("--crop-pad", type=float, default=0.18)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--label-smoothing", type=float, default=0.03)
    train_p.add_argument("--seed", type=int, default=7)
    train_p.add_argument("--workers", type=int, default=0)
    train_p.add_argument("--limit-samples", type=int, default=0)
    train_p.add_argument("--limit-valid", type=int, default=0)
    train_p.add_argument("--cpu", action="store_true")
    train_p.set_defaults(func=train)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--weights", required=True)
    pred_p.add_argument("--image", required=True)
    pred_p.add_argument("--threshold", type=float, default=0.5)
    pred_p.add_argument("--cpu", action="store_true")
    pred_p.set_defaults(func=predict)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
