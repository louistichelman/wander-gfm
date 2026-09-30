"""NBFNet link-prediction baseline for AnyGraph LP datasets."""

from .model import NBFNet
from .train import TrainConfig, train_nbfnet
from .eval import evaluate_nbfnet

__all__ = ["NBFNet", "TrainConfig", "train_nbfnet", "evaluate_nbfnet"]
