"""MLP structural causal model: features and labels without graph convolution.

Same causal setup as ``GNNSCM`` but with linear layers only. Used in
features-first generation, where the graph is built after ``X`` and ``y``.
"""

from __future__ import annotations

import math
import random
from functools import partial
from typing import Any, Dict

import torch
from torch import Tensor, nn

from .utils import GaussianNoise, XSampler


class MLPSCM(nn.Module):
    """Generates synthetic node-classification datasets using a pure MLP-based
    Structural Causal Model -- identical to :class:`GNNSCM` but without any
    graph convolution or structural features.

    Used in the *features-first* generation mode where the graph does not exist
    yet at dataset-generation time.

    Parameters
    ----------
    seq_len : int
        Number of nodes (samples) to generate.
    num_features : int
        Number of node features in the output ``X``.
    num_outputs : int
        Dimensionality of the target ``y`` per node.
    is_causal : bool
        If ``True``, ``X`` and ``y`` are sampled from concatenated intermediate
        hidden states.  If ``False``, initial causes become ``X`` and the final
        layer output becomes ``y``.
    num_causes : int
        Number of initial root cause variables.
    y_is_effect : bool
        How ``y`` is selected in causal mode.
    in_clique : bool
        Whether ``X`` and ``y`` are drawn from a contiguous block.
    sort_features : bool
        Whether to sort selected feature indices.
    num_layers : int
        Total number of layers (>= 2).
    hidden_dim : int
        Hidden representation dimensionality.
    mlp_activations : callable
        Activation function class.
    init_std : float
        Standard deviation for weight initialisation.
    block_wise_dropout : bool
        If ``True``, block-wise sparse init; otherwise Bernoulli dropout init.
    mlp_dropout_prob : float
        Dropout probability for standard init.
    scale_init_std_by_dropout : bool
        Whether to scale ``init_std`` to compensate for dropout.
    sampling : str
        Distribution for initial cause variables.
    pre_sample_cause_stats : bool
        Pre-sample per-cause mean/std.
    noise_std : float
        Gaussian noise std after each layer.
    pre_sample_noise_std : bool
        Per-dim noise std sampling.
    device : str
        Target device.
    **kwargs
        Unused HPs from parent config (ignored for drop-in compatibility).
    """

    def __init__(
        self,
        *,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features

        assert num_layers >= 2, "Number of layers must be at least 2."
        self.num_layers = num_layers

        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        if self.is_causal:
            self.hidden_dim = max(
                self.hidden_dim, self.num_outputs + 2 * self.num_features
            )
        else:
            self.num_causes = self.num_features

        self.xsampler = partial(
            XSampler,
            num_features=self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers -- plain Linear (no graph convolution)
        num_inputs = self.num_causes
        layers: list[nn.Module] = [nn.Linear(num_inputs, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self._make_layer())
        if not self.is_causal:
            layers.append(self._make_layer(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

        self._initialize_parameters()

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _make_layer(self, is_output_layer: bool = False) -> nn.Sequential:
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        linear = nn.Linear(self.hidden_dim, out_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std
        noise = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear, noise)

    # ------------------------------------------------------------------
    # Initialisation (identical to GNNSCM)
    # ------------------------------------------------------------------

    def _initialize_parameters(self):
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self._init_block_dropout(param, i)
            else:
                self._init_normal(param, i)

    def _init_block_dropout(self, param: Tensor, index: int):
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(
                param[block_slice],
                std=self.init_std
                / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1),
            )

    def _init_normal(self, param: Tensor, index: int):
        if param.dim() == 2:
            dropout_prob = self.mlp_dropout_prob if index > 0 else 0
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / (
                (1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1
            )
            nn.init.normal_(param, std=std)
            param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self):
        causes = self.xsampler(self.seq_len).sample()  # (seq_len, num_causes)

        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        # Skip first two entries (causes + first linear-only layer)
        outputs = outputs[2:]

        X, y = self._handle_outputs(causes, outputs)

        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X.cpu(), y.cpu()

    def _handle_outputs(self, causes: Tensor, outputs: list[Tensor]):
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if self.in_clique:
                start = random.randint(
                    0, outputs_flat.shape[-1] - self.num_outputs - self.num_features
                )
                random_perm = start + torch.randperm(
                    self.num_outputs + self.num_features, device=self.device
                )
            else:
                random_perm = torch.randperm(
                    outputs_flat.shape[-1] - 1, device=self.device
                )

            indices_X = random_perm[
                self.num_outputs : self.num_outputs + self.num_features
            ]
            if self.y_is_effect:
                indices_y = list(range(-self.num_outputs, 0))
            else:
                indices_y = random_perm[: self.num_outputs]

            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)

            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            X = causes
            y = outputs[-1]

        return X, y
