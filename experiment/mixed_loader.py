"""Mixed training loader composing a real-graph and a synthetic-graph loader."""

from typing import Iterator

import torch

from .query_loader import LoaderYield, MultiGraphLoader
from .synthetic_loader import SyntheticGraphLoader


class MixedTrainLoader:
    """Per-step mix between a real (``MultiGraphLoader``) and a synthetic
    (``SyntheticGraphLoader``) training loader.

    At each step a fair coin with probability ``p_real`` decides whether to
    pull from the real or the synthetic loader. The decision sequence is
    seeded from the epoch number so all DDP ranks make identical choices,
    keeping the underlying loaders in sync across ranks.

    The wrapped loaders are expected to be configured in training mode and
    to yield identically-shaped items (singletons or
    ``list[(name, batch, data)]`` whose length matches each loader's per-rank
    graphs-per-step (see DDP notes on ``MultiGraphLoader`` / ``SyntheticGraphLoader``).

    Parameters
    ----------
    real : MultiGraphLoader
        Training loader over real datasets.
    synthetic : SyntheticGraphLoader
        On-the-fly synthetic graph loader.
    p_real : float
        Probability in ``[0, 1]`` that a step is drawn from ``real``.
    batch_per_epoch : int
        Optimizer steps per epoch per rank (same value as ``--batch_per_epoch``;
        not divided by world size — DDP graph splitting is handled inside loaders).
    """

    def __init__(
        self,
        real: MultiGraphLoader,
        synthetic: SyntheticGraphLoader,
        p_real: float,
        batch_per_epoch: int,
    ):
        if not (0.0 <= p_real <= 1.0):
            raise ValueError(f"p_real must be in [0, 1], got {p_real}")
        self.real = real
        self.synthetic = synthetic
        self.p_real = p_real
        self.batch_per_epoch = batch_per_epoch
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.real.set_epoch(epoch)
        self.synthetic.set_epoch(epoch)

    def close(self, kill_workers: bool = True) -> None:
        closer = getattr(self.synthetic, "close", None)
        if callable(closer):
            closer(kill_workers=kill_workers)

    def __len__(self) -> int:
        return self.batch_per_epoch

    def __iter__(self) -> Iterator[LoaderYield]:
        g = torch.Generator()
        g.manual_seed(self._epoch)
        coin = (torch.rand(self.batch_per_epoch, generator=g) < self.p_real)
        n_real = int(coin.sum().item())
        n_synth = self.batch_per_epoch - n_real

        # Configure exact step counts so neither side over- or under-generates.
        # Both fields are interpreted as "yields per epoch" by the wrapped loaders.
        self.real.batch_per_epoch = n_real
        self.synthetic.batch_per_epoch = n_synth

        real_iter = iter(self.real) if n_real > 0 else iter(())
        synth_iter = iter(self.synthetic) if n_synth > 0 else iter(())

        for is_real in coin.tolist():
            if is_real:
                yield next(real_iter)
            else:
                yield next(synth_iter)
