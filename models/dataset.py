"""
Interaction Dataset
====================
Wraps raw (user_id, item_id) interaction pairs and generates negative samples
for two-tower model training.  For every observed positive interaction, N
negatives are sampled: random items the user has never interacted with,
labelled 0.

User and item string IDs are encoded to contiguous integers starting at 1;
index 0 is reserved as a padding index (matching padding_idx=0 in the model's
embedding layers).

Usage:
    df = pd.read_parquet("data/processed/interactions.parquet")
    dataset = InteractionDataset(df, n_negatives=4)

    # Pass to the model constructor
    model = TwoTowerModel(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
    )

    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    for user_ids, item_ids, labels in loader:
        loss = model.compute_loss(user_ids, item_ids, labels)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Args:
        interactions: DataFrame with at least ``user_id`` and ``item_id`` columns.
        n_negatives:  number of negative samples generated per positive interaction.
        seed:         random seed for reproducible negative sampling.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_negatives: int = 4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # --- Build string → integer encoders (1-indexed; 0 reserved for padding) ---
        unique_users = interactions["user_id"].unique()
        unique_items = interactions["item_id"].unique()

        self._user_enc: dict[str, int] = {u: i for i, u in enumerate(unique_users, start=1)}
        self._item_enc: dict[str, int] = {it: i for i, it in enumerate(unique_items, start=1)}
        self._num_users = len(unique_users)
        self._num_items = len(unique_items)

        # --- Encode positive pairs ---
        user_idx = interactions["user_id"].map(self._user_enc).to_numpy(np.int64)
        item_idx = interactions["item_id"].map(self._item_enc).to_numpy(np.int64)

        # --- Sample negatives ---
        # For each positive, repeat the user n_negatives times and draw random items.
        # Item indices are in [1, num_items] (1-indexed).  Rejection-resample any
        # (user, item) pair that already exists as a positive.  Given the extreme
        # sparsity of the interaction matrix (~0.006%) virtually no resampling occurs.
        pos_set = set(zip(user_idx.tolist(), item_idx.tolist()))

        neg_user_idx = np.repeat(user_idx, n_negatives)
        neg_item_idx = rng.integers(1, self._num_items + 1, size=len(neg_user_idx), dtype=np.int64)

        collision = np.array(
            [(int(u), int(it)) in pos_set for u, it in zip(neg_user_idx, neg_item_idx)]
        )
        while collision.any():
            bad = np.where(collision)[0]
            neg_item_idx[bad] = rng.integers(
                1, self._num_items + 1, size=len(bad), dtype=np.int64
            )
            collision[bad] = [
                (int(neg_user_idx[i]), int(neg_item_idx[i])) in pos_set for i in bad
            ]

        # --- Concatenate positives and negatives ---
        self._user_ids = np.concatenate([user_idx,                             neg_user_idx])
        self._item_ids = np.concatenate([item_idx,                             neg_item_idx])
        self._labels   = np.concatenate([np.ones(len(user_idx), dtype=np.float32),
                                         np.zeros(len(neg_user_idx), dtype=np.float32)])

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_users(self) -> int:
        """Vocabulary size for the user embedding table (excludes padding index 0)."""
        return self._num_users

    @property
    def num_items(self) -> int:
        """Vocabulary size for the item embedding table (excludes padding index 0)."""
        return self._num_items

    @property
    def user_enc(self) -> dict[str, int]:
        """Mapping from raw user_id string to integer index."""
        return self._user_enc

    @property
    def item_enc(self) -> dict[str, int]:
        """Mapping from raw item_id string to integer index."""
        return self._item_enc

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self._user_ids[idx], dtype=torch.long),
            torch.tensor(self._item_ids[idx], dtype=torch.long),
            torch.tensor(self._labels[idx],   dtype=torch.float32),
        )
