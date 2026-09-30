"""Loss functions and metrics for Wander."""

from typing import Dict
from torch import Tensor
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List
import numpy as np
from sklearn.metrics import roc_auc_score

from data.lp_rank_stats import is_per_relation_metric


def binary_roc_auc_score(y_true: np.ndarray, prob_positive: np.ndarray) -> float:
    """ROC AUC for binary labels using predicted probability of class 1.

    Returns ``nan`` when the score is undefined (e.g. only one class present).
    """
    try:
        return float(roc_auc_score(y_true, prob_positive))
    except ValueError:
        return float("nan")


def macro_ovr_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged one-vs-rest ROC AUC for classification scores.

    Ranking metric: does **not** require argmax / a 0.5 decision threshold.

    Args:
        y_true: ``[N]`` integer class labels.
        y_score: ``[N, C]`` per-class scores (typically softmax probabilities),
            or ``[N]`` positive-class scores for the legacy binary path.

    Returns:
        Mean OvR AUROC over classes that have both positives and negatives in
        ``y_true``. For ``C == 2`` this matches :func:`binary_roc_auc_score`
        on column 1. Returns ``nan`` if no class is scorable.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score)
    if y_true.shape[0] == 0:
        return float("nan")
    if y_score.ndim == 1:
        return binary_roc_auc_score(y_true, y_score)
    if y_score.ndim != 2:
        raise ValueError(
            f"y_score must be [N] or [N, C], got shape {tuple(y_score.shape)}"
        )
    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true/y_score length mismatch: {y_true.shape[0]} vs {y_score.shape[0]}"
        )
    num_classes = int(y_score.shape[1])
    if num_classes < 2:
        return float("nan")
    if num_classes == 2:
        return binary_roc_auc_score(y_true, y_score[:, 1])

    aucs: list[float] = []
    for c in range(num_classes):
        yt = (y_true == c).astype(np.int64, copy=False)
        if int(yt.min()) == int(yt.max()):
            continue
        try:
            aucs.append(float(roc_auc_score(yt, y_score[:, c])))
        except ValueError:
            continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def compute_link_prediction_loss(
    scores: Tensor,
    adversarial_temperature: float = 0.0,
    info_nce_temperature: float = 0.0,
) -> Tensor:
    """Compute loss when scores are already gathered for candidates.

    Used with models (e.g. Wander) that score only candidate tails directly.
    Applies binary cross-entropy with optional adversarial negative
    reweighting and an optional InfoNCE auxiliary term.

    Args:
        scores: [batch_size, 1 + num_negatives] where column 0 is the
            positive tail and columns 1: are negatives.
        adversarial_temperature: Temperature for adversarial negative
            reweighting via softmax over negative scores.  When <= 0
            negatives are weighted uniformly.
        info_nce_temperature: Temperature for an additive InfoNCE
            (softmax cross-entropy) loss term.  Disabled when <= 0.

    Returns:
        Scalar loss tensor.
    """
    target = torch.zeros_like(scores)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(
        scores, target, reduction="none"
    )
    neg_weight = torch.ones_like(scores)
    if adversarial_temperature > 0:
        with torch.no_grad():
            neg_weight[:, 1:] = F.softmax(
                scores[:, 1:] / adversarial_temperature, dim=-1
            )
    else:
        neg_weight[:, 1:] = 1 / (scores.shape[1] - 1)
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    loss = loss.mean()

    if info_nce_temperature > 0.0:
        info_nce_target = torch.zeros_like(scores[:, 0], dtype=torch.long)
        loss += F.cross_entropy(
            scores / info_nce_temperature, info_nce_target
        )
    return loss


def compute_node_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss for native node classification.

    Args:
        logits: [num_nodes, num_classes] unnormalized scores.
        targets: [num_nodes] integer class labels.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, targets)


def aggregate_metrics(per_graph_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-graph metrics into overall metrics.
    
    Args:
        per_graph_metrics: Dict mapping graph names to metric dicts.
        
    Returns:
        Aggregated metrics (averaged across graphs).
    """
    all_metrics = defaultdict(list)
    
    for graph_name, metrics in per_graph_metrics.items():
        if graph_name == "aggregate":
            continue
        for metric_name, value in metrics.items():
            # Per-relation breakdowns are dataset-local: relation id 95 means a
            # different relation in every graph, so averaging them is nonsense.
            if is_per_relation_metric(metric_name):
                continue
            all_metrics[metric_name].append(value)
    
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}

def aggregate_seed_results(
        results_list: List[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate metrics across seeds.
        
        Args:
            results_list: List of per-graph metric dicts from each seed.
            split: "val" or "test".
            
        Returns:
            Dict with mean and std for each metric.
        """
        aggregated = defaultdict(lambda: defaultdict(list))
        
        for result in results_list:
            for graph_name, metrics in result.items():
                for metric_name, value in metrics.items():
                    aggregated[graph_name][metric_name].append(value)
        
        # Compute mean and std
        final = {}
        for graph_name, metrics in aggregated.items():
            final[graph_name] = {}
            for metric_name, values in metrics.items():
                final[graph_name][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
        
        return final
    
def print_aggregated_results(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
    split: str,
):
    """Print aggregated results in a nice format."""
    print(f"\n{split.upper()} Results:")
    for graph_name, metrics in aggregated.items():
        print(f"  {graph_name}:")
        for metric_name, stats in metrics.items():
            print(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
