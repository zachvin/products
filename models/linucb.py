"""
LinUCB Contextual Bandit
========================
Disjoint LinUCB with a single shared parameter vector.  The context for each
(user, item) pair is the concatenation of their feature vectors.

The UCB score for context x is:

    score = θᵀx + α · √(xᵀ A⁻¹ x)

where θ = A⁻¹b, A is updated via Sherman-Morrison to avoid O(d³) inversions,
and α trades off exploration against exploitation.

Usage:
    model = LinUCB(context_dim=90, alpha=1.0)

    # Score a single pair
    s = model.score(user_vec, item_vec)

    # Rank a set of items for one user — returns indices sorted best-first
    order = model.rank(user_vec, item_matrix)   # item_matrix: (n_items, d_item)

    # Update after observing a reward (e.g. click=1, no-click=0)
    model.update(user_vec, item_vec, reward=1.0)

    model.save("feature_store/linucb.pkl")
    model = LinUCB.load("feature_store/linucb.pkl")
"""

import pickle
from pathlib import Path

import numpy as np


class LinUCB:
    """
    Args:
        context_dim: dimensionality of concat(user_features, item_features).
        alpha:       exploration coefficient — higher values explore more.
    """

    def __init__(self, context_dim: int, alpha: float = 1.0) -> None:
        self.context_dim = context_dim
        self.alpha = alpha
        self._A_inv = np.eye(context_dim, dtype=np.float64)
        self._b = np.zeros(context_dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, user_features: np.ndarray, item_features: np.ndarray) -> float:
        """UCB score for a single (user, item) pair."""
        x = self._context(user_features, item_features)
        return float(self._ucb_scores(x.reshape(1, -1))[0])

    def rank(self, user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
        """
        Rank items for one user by descending UCB score.

        Args:
            user_features: (d_user,) feature vector for the user.
            item_features: (n_items, d_item) feature matrix for candidate items.

        Returns:
            Integer indices into item_features sorted best-first.
        """
        n = len(item_features)
        X = np.concatenate(
            [np.tile(user_features, (n, 1)), item_features], axis=1
        )
        scores = self._ucb_scores(X)
        return np.argsort(-scores)

    def update(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        reward: float,
    ) -> None:
        """
        Update model parameters after observing a reward.

        Args:
            user_features: (d_user,) context for the user.
            item_features: (d_item,) context for the chosen item.
            reward:        observed reward (e.g. 1.0 for click, 0.0 for no-click).
        """
        x = self._context(user_features, item_features)
        # Sherman-Morrison rank-1 update of A⁻¹
        Ax = self._A_inv @ x
        self._A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self._b += reward * x

    def save(self, path: str | Path) -> None:
        """Persist the model to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "context_dim": self.context_dim,
                    "alpha": self.alpha,
                    "A_inv": self._A_inv,
                    "b": self._b,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LinUCB":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(context_dim=data["context_dim"], alpha=data["alpha"])
        model._A_inv = data["A_inv"]
        model._b = data["b"]
        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _context(self, user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
        return np.concatenate([user_features, item_features]).astype(np.float64)

    def _ucb_scores(self, X: np.ndarray) -> np.ndarray:
        """Vectorised UCB score for a batch of context rows X (n, d)."""
        theta = self._A_inv @ self._b
        exploit = X @ theta
        XA = X @ self._A_inv
        explore = self.alpha * np.sqrt((XA * X).sum(axis=1))
        return exploit + explore
