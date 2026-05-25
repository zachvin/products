"""
Two-Tower Recommendation Model
================================
Each tower maps an ID to a unit-norm vector via an embedding lookup followed
by an MLP.  The dot product of two unit vectors equals their cosine similarity,
so the model learns to place interacted (user, item) pairs close together in
the shared embedding space.

Training uses BCE loss with binary interaction labels (1 = interacted, 0 = not).
The dot product is treated as a logit and sigmoid is applied internally by
BCEWithLogitsLoss, which is numerically more stable than applying sigmoid first.

Usage:
    model = TwoTowerModel(num_users=50_000, num_items=20_000)

    # Training step
    loss = model.compute_loss(user_ids, item_ids, labels)

    # Inference — encode each side independently
    user_vecs = model.encode_users(user_ids)   # shape (B, output_dim)
    item_vecs = model.encode_items(item_ids)   # shape (B, output_dim)
    scores    = (user_vecs * item_vecs).sum(-1) # cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Tower(nn.Module):
    """Embedding lookup → MLP → L2-normalised output vector."""

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # +1 so indices 1..num_entities are all valid; 0 is the padding row.
        self.embedding = nn.Embedding(num_entities + 1, embedding_dim, padding_idx=0)

        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Args:
        num_users:     total number of users (vocabulary size for user embeddings)
        num_items:     total number of items (vocabulary size for item embeddings)
        embedding_dim: dimension of the initial ID embedding for each tower
        hidden_dims:   MLP hidden layer widths, applied identically to both towers
        output_dim:    dimension of the final unit-norm vector from each tower
        dropout:       dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.user_tower = _Tower(num_users, embedding_dim, hidden_dims, output_dim, dropout)
        self.item_tower = _Tower(num_items, embedding_dim, hidden_dims, output_dim, dropout)
        self._loss_fn   = nn.BCEWithLogitsLoss()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return per-pair cosine similarity scores shaped (batch,).

        These are logits — apply sigmoid to convert to probabilities.
        """
        user_vecs = self.user_tower(user_ids)
        item_vecs = self.item_tower(item_ids)
        return (user_vecs * item_vecs).sum(dim=-1)

    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss between cosine similarity logits and binary interaction labels.

        Args:
            user_ids: (batch,) integer user indices
            item_ids: (batch,) integer item indices
            labels:   (batch,) float or bool — 1 for observed interactions, 0 for negatives
        """
        scores = self.forward(user_ids, item_ids)
        return self._loss_fn(scores, labels.float())

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised user vectors shaped (batch, output_dim)."""
        return self.user_tower(user_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised item vectors shaped (batch, output_dim)."""
        return self.item_tower(item_ids)
