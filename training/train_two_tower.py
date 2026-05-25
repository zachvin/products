"""
Two-Tower Training Loop
========================
Trains the two-tower recommendation model on interaction data and logs
metrics, parameters, and the final model artifact to MLflow.

Run:
    python training/train_two_tower.py

Checkpoints are written to ./training/checkpoints/ after each epoch.
The best checkpoint (lowest validation loss) is logged to MLflow as an artifact.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.dataset import InteractionDataset
from models.two_tower import TwoTowerModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent.parent
_DATA_PATH  = _ROOT / "data" / "processed" / "interactions.parquet"
_CKPT_DIR   = _ROOT / "training" / "checkpoints"
_MLFLOW_URI = _ROOT / "mlflow"

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    n_negatives   = 4,
    embedding_dim = 64,
    hidden_dims   = "128,64",
    output_dim    = 64,
    dropout       = 0.1,
    lr            = 1e-3,
    weight_decay  = 1e-5,
    batch_size    = 2048,
    epochs        = 10,
    val_fraction  = 0.1,
    seed          = 42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the two-tower recommendation model.")
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(val), default=val)
    return p.parse_args()


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    cut = int(n * val_fraction)
    return idx[cut:], idx[:cut]  # train, val


def _run_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """One full pass through a DataLoader. Returns mean loss.

    If optimizer is None the pass is run in eval mode with no gradient tracking.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for user_ids, item_ids, labels in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels   = labels.to(device)

            loss = model.compute_loss(user_ids, item_ids, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n      = labels.size(0)
            total_loss   += loss.item() * batch_n
            total_samples += batch_n

    return total_loss / total_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args = _parse_args()
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading interactions...")
    interactions = pd.read_parquet(_DATA_PATH)
    print(f"  {len(interactions):,} interactions")

    print(f"Building dataset (n_negatives={args.n_negatives})...")
    dataset = InteractionDataset(interactions, n_negatives=args.n_negatives, seed=args.seed)
    print(f"  {len(dataset):,} total samples")
    print(f"  {dataset.num_users:,} users  |  {dataset.num_items:,} items")

    train_idx, val_idx = _split_indices(len(dataset), args.val_fraction, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # --- Model ---
    model = TwoTowerModel(
        num_users     = dataset.num_users,
        num_items     = dataset.num_items,
        embedding_dim = args.embedding_dim,
        hidden_dims   = hidden_dims,
        output_dim    = args.output_dim,
        dropout       = args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # --- MLflow ---
    mlflow.set_tracking_uri(_MLFLOW_URI.as_uri())
    mlflow.set_experiment("two-tower")

    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss  = float("inf")
    best_ckpt_path = _CKPT_DIR / "best.pt"

    with mlflow.start_run():
        mlflow.log_params({
            "n_negatives"   : args.n_negatives,
            "embedding_dim" : args.embedding_dim,
            "hidden_dims"   : args.hidden_dims,
            "output_dim"    : args.output_dim,
            "dropout"       : args.dropout,
            "lr"            : args.lr,
            "weight_decay"  : args.weight_decay,
            "batch_size"    : args.batch_size,
            "epochs"        : args.epochs,
            "val_fraction"  : args.val_fraction,
            "num_users"     : dataset.num_users,
            "num_items"     : dataset.num_items,
        })

        for epoch in range(1, args.epochs + 1):
            train_loss = _run_epoch(model, train_loader, optimizer, device)
            val_loss   = _run_epoch(model, val_loader,   None,      device)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={current_lr:.2e}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr},
                step=epoch,
            )

            # Save checkpoint each epoch; keep the best separately
            ckpt_path = _CKPT_DIR / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss"   : val_loss,
                    "user_enc"   : dataset.user_enc,
                    "item_enc"   : dataset.item_enc,
                },
                ckpt_path,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(torch.load(ckpt_path, weights_only=False), best_ckpt_path)
                print(f"  ✓ new best val_loss={best_val_loss:.4f}")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(str(best_ckpt_path))
        print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
        print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
