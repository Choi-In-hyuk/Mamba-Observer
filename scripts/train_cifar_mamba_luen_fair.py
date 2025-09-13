#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import csv
import time
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from tqdm import tqdm
import numpy as np

# Import the custom Mamba with Luenberger Observer
# Assumes you have: from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver
from mamba_ssm.modules.mamba_luen import MambaBlockWithObserver


# ----------------------------
# Model definition
# ----------------------------
class SimpleMambaClassifierLuenberger(nn.Module):
    """Simplified Mamba classifier with Luenberger Observer for CIFAR-10."""
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        observer_alpha: float = 0.2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # Project per-pixel RGB to d_model (sequence length = 32*32)
        self.input_proj = nn.Linear(3, d_model)

        self.mamba_layers = MambaBlockWithObserver(
            num_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_observer=True,
            observer_alpha=observer_alpha,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        b = x.shape[0]
        # (B, 32, 32, 3) -> (B, 1024, 3)
        x = x.permute(0, 2, 3, 1).reshape(b, -1, 3)
        x = self.input_proj(x)          # (B, 1024, d_model)
        x = self.mamba_layers(x)        # (B, 1024, d_model)
        x = x.mean(dim=1)               # global average over sequence -> (B, d_model)
        x = self.norm(x)
        return self.classifier(x)       # (B, num_classes)


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism trade-offs; enable as needed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stratified_train_val_split(dataset: CIFAR10, val_size: int = 5000, seed: int = 0) -> Tuple[Subset, Subset]:
    """Create a stratified split (45k train / 5k val) preserving class distribution."""
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    assert val_size % num_classes == 0, "val_size should be divisible by number of classes (10 for CIFAR-10)."

    per_class_val = val_size // num_classes
    train_indices, val_indices = [], []

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        val_idx_c = idx_c[:per_class_val]
        train_idx_c = idx_c[per_class_val:]
        val_indices.extend(val_idx_c.tolist())
        train_indices.extend(train_idx_c.tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.collected = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module):
        self.collected = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collected[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.collected:
                param.data = self.collected[name].clone()
        self.collected = {}


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler = None,
    max_grad_norm: float = 1.0,
    ema: EMA = None,
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Reset observer states per mini-batch (as requested)
        if hasattr(model, "mamba_layers") and hasattr(model.mamba_layers, "reset_observers"):
            model.mamba_layers.reset_observers()

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(total_acc / n_batches) * 100:.2f}%")

    return total_loss / max(n_batches, 1), (total_acc / max(n_batches, 1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_ema: bool = False,
    ema: EMA = None,
    desc: str = "Eval",
):
    model.eval()
    if use_ema and ema is not None:
        ema.apply_shadow(model)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Reset observer states for evaluation
        if hasattr(model, "mamba_layers") and hasattr(model.mamba_layers, "reset_observers"):
            model.mamba_layers.reset_observers()

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1

    if use_ema and ema is not None:
        ema.restore(model)

    return total_loss / max(n_batches, 1), (total_acc / max(n_batches, 1))


# ----------------------------
# Main training procedure (per seed)
# ----------------------------
def run_one_seed(args, seed: int) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    root = Path(args.data_root)
    train_full = CIFAR10(root=str(root), train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=str(root), train=False, download=True, transform=transform_test)

    # Stratified split: 45k train / 5k val
    train_set, val_set = stratified_train_val_split(train_full, val_size=args.val_size, seed=seed)

    # DataLoaders (use a fixed generator for deterministic shuffling)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=g,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    # Model
    model = SimpleMambaClassifierLuenberger(
        num_classes=10,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        observer_alpha=args.observer_alpha,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    ).to(device)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    # Optimizer / Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine with warmup
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    args.max_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    if args.label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # Checkpoint paths
    seed_dir = Path(args.out_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_path = seed_dir / "best.pth"
    last_path = seed_dir / "last.pth"

    # Logs
    print("=" * 60)
    print(f"Seed {seed} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Observer alpha: {args.observer_alpha}")
    print(f"AMP: {args.amp} | EMA: {args.use_ema} (decay={args.ema_decay})")
    print("=" * 60)

    best_val_acc = 0.0
    best_test_at_best_val = 0.0
    best_epoch = -1

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=args.amp, scaler=scaler, max_grad_norm=args.clip_grad_norm, ema=ema
        )

        # Scheduler steps per-iteration already handled by Lambda via global_step below
        # If you prefer per-epoch, call scheduler.step() here.

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            use_ema=args.eval_with_ema, ema=ema, desc="Val"
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device,
            use_ema=args.eval_with_ema, ema=ema, desc="Test"
        )

        epoch_time = time.time() - t0

        # Save best by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_at_best_val = test_acc
            best_epoch = epoch
            save_checkpoint(best_path, {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "test_acc_at_best_val": best_test_at_best_val,
                "args": vars(args),
                "seed": seed,
            })
            print(f"[Seed {seed}] New best @ epoch {epoch}: "
                  f"Val {best_val_acc*100:.2f}% | Test {best_test_at_best_val*100:.2f}% -> {best_path}")

        # Save last
        save_checkpoint(last_path, {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "test_acc_at_best_val": best_test_at_best_val,
            "args": vars(args),
            "seed": seed,
        })

        # Progress print
        print(f"Epoch {epoch:03d} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | "
              f"Test Loss {test_loss:.4f} Acc {test_acc*100:.2f}% | "
              f"Best Val {best_val_acc*100:.2f}% (Test@Best {best_test_at_best_val*100:.2f}%) | "
              f"Time {epoch_time:.1f}s | LR {scheduler.get_last_lr()[0]:.6f}")

        # Step scheduler per-iteration equivalent
        steps_this_epoch = len(train_loader)
        for _ in range(steps_this_epoch):
            scheduler.step()
            global_step += 1

    return {
        "seed": seed,
        "best_val_acc": best_val_acc * 100.0,
        "test_acc_at_best_val": best_test_at_best_val * 100.0,
        "best_epoch": best_epoch,
    }


# ----------------------------
# Argparse
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="CIFAR-10 Mamba + Luenberger Observer (SOTA-ready)")

    # Data / IO
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="checkpoint_cifar_mamba_luen_sota")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-size", type=int, default=5000)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)

    # Scheduler
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0)  # set internally after dataloader build

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--observer-alpha", type=float, default=0.2)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)

    # AMP / EMA
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--eval-with-ema", action="store_true",
                   help="Evaluate using EMA weights (common in SOTA reporting).")
    p.add_argument("--ema-decay", type=float, default=0.9999)

    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    return p


# ----------------------------
# Entry
# ----------------------------
def main():
    args = build_argparser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in args.seeds:
        res = run_one_seed(args, seed)
        all_results.append(res)

    # Aggregate
    test_scores = [r["test_acc_at_best_val"] for r in all_results]
    val_scores = [r["best_val_acc"] for r in all_results]

    test_mean = float(np.mean(test_scores))
    test_std = float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0
    val_mean = float(np.mean(val_scores))
    val_std = float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0

    # Save CSV
    csv_path = Path(args.out_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "best_val_acc(%)", "test_acc_at_best_val(%)", "best_epoch"])
        for r in all_results:
            writer.writerow([r["seed"], f"{r['best_val_acc']:.2f}", f"{r['test_acc_at_best_val']:.2f}", r["best_epoch"]])
        writer.writerow([])
        writer.writerow(["val_mean(%)", f"{val_mean:.2f}"])
        writer.writerow(["val_std(%)", f"{val_std:.2f}"])
        writer.writerow(["test_mean(%)", f"{test_mean:.2f}"])
        writer.writerow(["test_std(%)", f"{test_std:.2f}"])

    # Final print
    print("=" * 60)
    print("FINAL RESULTS (Best-Val checkpoint -> Test)")
    for r in all_results:
        print(f"Seed {r['seed']}: Val {r['best_val_acc']:.2f}% | Test {r['test_acc_at_best_val']:.2f}% | Best epoch {r['best_epoch']}")
    print("-" * 60)
    print(f"VAL:  mean {val_mean:.2f}%  std {val_std:.2f}%")
    print(f"TEST: mean {test_mean:.2f}%  std {test_std:.2f}%")
    print(f"Saved CSV -> {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
