"""BUDDY link-prediction baseline for AnyGraph homogeneous and bipartite LP."""

from .eval import evaluate_recall_at_k, score_pairs
from .model import BUDDY
from .train import TrainConfig, train_buddy

__all__ = ["BUDDY", "TrainConfig", "evaluate_recall_at_k", "score_pairs", "train_buddy"]
