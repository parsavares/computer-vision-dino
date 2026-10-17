#!/usr/bin/env python
import os
import argparse
import random
import time

import numpy as np
import pandas as pd
import cv2

from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm


# -------------------------
# Utils
# -------------------------

def process_labels(labels_dir: str, split: str) -> pd.DataFrame:
    """
    Load CSV with annotations for given split.
    Expected path: <labels_dir>/<split>.csv
    e.g. train.csv or val.csv
    """
    path = os.path.join(labels_dir, f"{split}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    labels = pd.read_csv(path)
    return labels


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset for segmentation
# -------------------------

class SPARKSegDataset(Dataset):
    """
    Segmentation dataset for SPARK:
    - Input: image
    - Target: per-pixel labels {0: background, 1: body (red), 2: solar panels (blue)}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
    ):
        self.root_dir = os.path.join(root_dir)
        self.labels = process_labels(self.root_dir, split)
        self.split = split
        self.image_size = image_size

        # ImageNet / ViT style normalization
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.labels)

    def _load_image(self, path: str):
        """
        Load image as RGB using OpenCV to avoid imageio/skimage backend issues.
        """
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # H, W, 3 in BGR
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image file: {path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # -> RGB
        return img

    def _load_mask(self, path: str, h_fallback: int, w_fallback: int):
        """
        Load mask as RGB using OpenCV.
        Fallback to all-background if missing or unreadable.
        """
        if os.path.isfile(path):
            mask_bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
            if mask_bgr is None:
                # broken file -> all background
                mask_rgb = np.zeros((h_fallback, w_fallback, 3), dtype=np.uint8)
            else:
                mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        else:
            # no mask found: all background
            mask_rgb = np.zeros((h_fallback, w_fallback, 3), dtype=np.uint8)

        return mask_rgb

    def _rgb_to_part_labels(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class indices:
        0 = background
        1 = body (red-ish)
        2 = solar panels (blue-ish)
        """
        r = mask_rgb[..., 0].astype(np.float32)
        g = mask_rgb[..., 1].astype(np.float32)
        b = mask_rgb[..., 2].astype(np.float32)

        H, W = r.shape
        labels = np.zeros((H, W), dtype=np.uint8)

        # heuristics for "red" and "blue" pixels
        body = (r > 128) & (g < 80) & (b < 80)
        panels = (b > 128) & (r < 80) & (g < 80)

        labels[body] = 1
        labels[panels] = 2

        return labels

    def __getitem__(self, idx: int):
        row = self.labels.iloc[idx]
        sat_name = row["Class"]
        img_name = row["Image name"]
        mask_name = row["Mask name"]
        _bbox = literal_eval(row["Bounding box"])  # not used for segmentation

        # Paths
        image_file = os.path.join(
            self.root_dir, "images", sat_name, self.split, img_name
        )
        mask_file = os.path.join(
            self.root_dir, "mask", sat_name, self.split, mask_name
        )

        # --- Load image ---
        image = self._load_image(image_file)
        h, w = image.shape[:2]

        # --- Load mask as RGB (for color-based labels) ---
        mask_rgb = self._load_mask(mask_file, h_fallback=h, w_fallback=w)

        # ---- Simple train-time augmentation (horizontal flip) ----
        if self.split == "train":
            if np.random.rand() < 0.5:
                # flip image and mask together
                image = np.ascontiguousarray(np.fliplr(image))
                mask_rgb = np.ascontiguousarray(np.fliplr(mask_rgb))

        # Resize both image and mask to common size
        img_resized = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_resized_rgb = cv2.resize(
            mask_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert mask RGB -> {0,1,2}
        seg_labels = self._rgb_to_part_labels(mask_resized_rgb)

        # Image to tensor + normalize
        img_tensor = transforms.ToTensor()(img_resized)  # [3, H, W], [0,1]
        img_tensor = self.normalize(img_tensor)

        # Segmentation mask to tensor (no normalization)
        mask_tensor = torch.from_numpy(seg_labels).long()  # [H, W]

        sample = {
            "img": img_tensor,
            "mask": mask_tensor,
        }
        return sample


# -------------------------
# DINO-based segmentation model
# -------------------------

class DinoSegModel(nn.Module):
    """
    DINO ViT encoder + simple upsampling decoder for segmentation.
    - Encoder: vit_small_patch16_224.dino (frozen, features_only)
    - Decoder: small conv head + bilinear upsample to image resolution
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # DINO ViT small as feature extractor
        self.encoder = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            features_only=True,
            out_indices=(3,),  # last stage feature map
        )
        enc_channels = self.encoder.feature_info.channels()[-1]

        # Simple decoder: convs + 1x1 projection -> num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(enc_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        # Freeze encoder (only train decoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] (e.g. 224 x 224)
        feats_list = self.encoder(x)      # list, take last map
        feat = feats_list[-1]             # [B, C, H', W']

        logits = self.decoder(feat)       # [B, num_classes, H', W']

        # Upsample to match input spatial size
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


# -------------------------
# Training / Evaluation
# -------------------------

def train_one_epoch_seg(
    model,
    loader,
    optimizer,
    device,
    criterion,
):
    model.train()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)  # [B, H, W]

        optimizer.zero_grad()

        logits = model(imgs)              # [B, C, H, W]
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)  # [B, H, W]
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

        running_loss += loss.item() * imgs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels
    return {"loss": avg_loss, "pixel_acc": pixel_acc}


@torch.no_grad()
def evaluate_seg(
    model,
    loader,
    device,
    criterion,
    num_classes: int = 3,
):
    model.eval()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    # IoU accumulators
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)  # [B, H, W]

        logits = model(imgs)
        loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)  # [B, H, W]

        running_loss += loss.item() * imgs.size(0)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

        # IoU per class
        for cls in range(num_classes):
            pred_c = (preds == cls)
            mask_c = (masks == cls)
            inter[cls] += (pred_c & mask_c).sum().item()
            union[cls] += (pred_c | mask_c).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels

    ious = []
    for cls in range(num_classes):
        if union[cls] > 0:
            ious.append(inter[cls] / union[cls])
        else:
            ious.append(0.0)
    mean_iou = float(np.mean(ious))

    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "mIoU": mean_iou,
        "IoU_per_class": ious,
    }


# -------------------------
# Argparse & main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINO-based segmentation model on SPARK dataset"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="spark-2024-train-val",
        help="Path to SPARK dataset root (contains images/, mask/, train.csv, val.csv, ...)",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Train split name (default: train). Validation will use split 'val'.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=60000,
        help="Number of samples to use from the TRAIN split (<= total train size).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (square) for DINO backbone (ViT patch16 = 224).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (segmentation is heavier than detection).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for decoder head.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_dino_seg_full",
        help="Directory to save checkpoints & logs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (best_model.pt or final_model.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Datasets: 60k train (train.csv) and 20k val (val.csv)
    # -------------------------
    print("Loading segmentation datasets...")

    train_dataset_full = SPARKSegDataset(
        root_dir=args.data_root,
        split=args.train_split,      # usually "train"
        image_size=args.image_size,
    )
    val_dataset_full = SPARKSegDataset(
        root_dir=args.data_root,
        split="val",                 # official validation split
        image_size=args.image_size,
    )

    train_total = len(train_dataset_full)
    val_total = len(val_dataset_full)
    print(f"Train split '{args.train_split}' size: {train_total}")
    print(f"Val split 'val' size:             {val_total}")

    # Optionally subsample train (default: full 60k)
    train_subset_size = min(args.subset_size, train_total)
    if train_subset_size < train_total:
        indices = list(range(train_total))
        random.shuffle(indices)
        train_indices = indices[:train_subset_size]
        train_dataset = Subset(train_dataset_full, train_indices)
    else:
        train_dataset = train_dataset_full

    # Validation: always use FULL val split (20k) for fair comparison
    val_dataset = val_dataset_full

    print(f"Final train size: {len(train_dataset)}")
    print(f"Final val size:   {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    num_seg_classes = 3  # 0=bg, 1=body, 2=panels
    model = DinoSegModel(num_classes=num_seg_classes).to(device)

    # Down-weight background to reduce "all background" bias
    class_weights = torch.tensor([0.1, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.Adam(
        decoder_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # Parameter counts (just for info)
    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = total_params - enc_params
    print(f"Total parameters:   {total_params:,}")
    print(f"Encoder parameters: {enc_params:,}")
    print(f"Decoder parameters: {dec_params:,}")

    # -------------------------
    # Training loop
    # -------------------------
    print("Starting segmentation training...")
    best_val_loss = float("inf")
    best_mIoU = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    start_epoch = 1

    # ----------------- RESUME SETUP -----------------
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # restore best metrics if present
            best_val_loss = ckpt.get("val_loss", best_val_loss)
            best_mIoU = ckpt.get("best_mIoU", best_mIoU)

            prev_epoch = ckpt.get("epoch", 0)
            start_epoch = prev_epoch + 1
            print(
                f"  -> Resuming from epoch {prev_epoch}, "
                f"continuing at epoch {start_epoch} "
                f"(best_mIoU so far: {best_mIoU:.4f})"
            )
        else:
            print(f"WARNING: --resume path '{args.resume}' not found, starting from scratch.")
    # ------------------------------------------------

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_stats = train_one_epoch_seg(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
        )
        val_stats = evaluate_seg(
            model,
            val_loader,
            device,
            criterion,
            num_classes=num_seg_classes,
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} - {elapsed:.1f}s\n"
            f"  Train: loss {train_stats['loss']:.4f}, "
            f"pixel acc {train_stats['pixel_acc']*100:.2f}%\n"
            f"  Val:   loss {val_stats['loss']:.4f}, "
            f"pixel acc {val_stats['pixel_acc']*100:.2f}%, "
            f"mIoU {val_stats['mIoU']:.4f}"
        )

        # ✅ choose best model by highest mIoU (primary metric)
        if val_stats["mIoU"] > best_mIoU:
            best_mIoU = val_stats["mIoU"]
            best_val_loss = val_stats["loss"]  # keep around for logging
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "best_mIoU": best_mIoU,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(
                f"  -> New best segmentation model saved to {best_ckpt_path} "
                f"(mIoU={best_mIoU:.4f})"
            )

    final_ckpt_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
            "best_mIoU": best_mIoU,
            "args": vars(args),
        },
        final_ckpt_path,
    )
    print(
        f"Segmentation training finished. Final model saved to {final_ckpt_path} "
        f"(best_mIoU={best_mIoU:.4f})"
    )


if __name__ == "__main__":
    main()
