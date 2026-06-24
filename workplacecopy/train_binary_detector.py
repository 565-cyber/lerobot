"""
Train a single binary ResNet18 detector.

Usage:
  python workplace/train_binary_detector.py --detector detector_yellow_on_table
  python workplace/train_binary_detector.py --detector detector_blue_on_table
  python workplace/train_binary_detector.py --detector detector_lid_open
  python workplace/train_binary_detector.py --detector all   # train all 3
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/zjq/lerobot/workplace/binary_detector_dataset")
SAVE_DIR = Path("/home/zjq/lerobot/workplace")

DETECTOR_NAMES = [
    "detector_yellow_on_table",
    "detector_blue_on_table",
    "detector_lid_open",
]

DETECTOR_LABELS = {
    "detector_yellow_on_table": "yellow_on_table",
    "detector_blue_on_table": "blue_on_table",
    "detector_lid_open": "lid_open",
}

DETECTOR_CKPTS = {
    "detector_yellow_on_table": "paper_detector_yellow_on_table.pt",
    "detector_blue_on_table": "paper_detector_blue_on_table.pt",
    "detector_lid_open": "paper_detector_lid_open.pt",
}

# ── Hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """Compute binary accuracy, precision, recall, F1."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    labels = labels.view(-1)
    preds = preds.view(-1)

    tp = (preds * labels).sum().float()
    fp = (preds * (1 - labels)).sum().float()
    fn = ((1 - preds) * labels).sum().float()
    tn = ((1 - preds) * (1 - labels)).sum().float()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "acc": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train_one_detector(detector_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_name = DETECTOR_LABELS[detector_name]
    save_path = SAVE_DIR / DETECTOR_CKPTS[detector_name]

    print("=" * 60)
    print(f"Training: {detector_name} ({label_name})")
    print(f"Device: {device}")
    print(f"Save to: {save_path}")
    print("=" * 60)

    dataset_dir = DATA_DIR / detector_name

    train_set = datasets.ImageFolder(str(dataset_dir / "train"), transform=get_transforms(True))
    val_set = datasets.ImageFolder(str(dataset_dir / "val"), transform=get_transforms(False))

    print(f"class_to_idx: {train_set.class_to_idx}")
    print(f"train: {len(train_set)}  val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples

        # ── Val ──
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_logits, all_labels)

        print(
            f"epoch={epoch:02d}/{EPOCHS}, "
            f"loss={avg_loss:.4f}, "
            f"val_acc={metrics['acc']:.4f}, "
            f"val_f1={metrics['f1']:.4f}, "
            f"val_prec={metrics['precision']:.4f}, "
            f"val_rec={metrics['recall']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "label_name": label_name,
                    "detector_name": detector_name,
                    "class_to_idx": train_set.class_to_idx,
                    "val_acc": metrics["acc"],
                    "val_f1": metrics["f1"],
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"  -> Saved best (f1={best_f1:.4f}): {save_path}")

    print(f"Done. Best val_f1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=str, required=True,
                        choices=["all"] + DETECTOR_NAMES,
                        help="Which detector to train, or 'all'")
    args = parser.parse_args()

    if args.detector == "all":
        for name in DETECTOR_NAMES:
            train_one_detector(name)
    else:
        train_one_detector(args.detector)


if __name__ == "__main__":
    main()
