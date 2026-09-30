"""GNN structural causal model: node features and labels via graph convolution.

Used in graph-first ``SCMPrior`` sampling. Initial causes (optionally plus
structural features) pass through ``SemiGraphConv`` layers; ``X`` and ``y``
are taken from intermediate or final activations.
"""

from __future__ import annotations

import math
import random
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
from scipy.sparse import coo_matrix, eye as sp_eye
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import degree

from .utils import GaussianNoise, SemiGraphConv, XSampler


@torch.no_grad()
def compute_pagerank(
    edge_index: Tensor,
    num_nodes: int,
    alpha=0.85,
    max_iterations=100,
    tol=1e-6,
    train_index=None,
    log: bool = True,
):
    assert alpha >= 0.5, "Please make sure that you provde alpha, not 1-alpha"
    device = edge_index.device
    src, dst = edge_index

    pv = torch.ones(num_nodes, device=device) / num_nodes
    degrees = degree(src, num_nodes=num_nodes).float()

    if train_index is None:
        reset_prob = (1 - alpha) / num_nodes
    else:
        reset_prob = torch.zeros(num_nodes, device=device)
        reset_prob[train_index] = 1
        reset_prob /= reset_prob.sum()
        reset_prob *= 1 - alpha

    for _ in range(max_iterations):
        prev_pv = pv.clone()

        messages = (pv / degrees.clamp(min=1.0))[src]
        new_pv = torch.zeros(num_nodes, device=device)
        new_pv.scatter_add_(0, dst, messages)
        pv = new_pv

        pv = alpha * pv + reset_prob

        err = torch.abs(pv - prev_pv).sum()
        if err < tol:
            break

    if log:
        pv = torch.log(tol + pv)
    return pv[..., None]


def _compute_laplacian_pe(edge_index: Tensor, num_nodes: int, k: int, device: torch.device) -> Tensor:
    """Compute the k smallest nontrivial Laplacian eigenvectors."""
    src, dst = edge_index.cpu().numpy()
    n = num_nodes

    adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])

    from scipy.sparse import diags
    D_inv_sqrt = diags(deg_inv_sqrt)
    L_norm = sp_eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    num_eigenvectors = min(k + 1, n - 1)
    if num_eigenvectors < 2:
        return torch.randn(n, k, device=device)

    eigenvalues, eigenvectors = eigsh(L_norm.tocsc(), k=num_eigenvectors, which="SM", tol=1e-5)

    # Drop the trivial (constant) eigenvector (eigenvalue ~0)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx[1: k + 1]]

    # Pad if fewer than k nontrivial eigenvectors
    if eigenvectors.shape[1] < k:
        pad = np.zeros((n, k - eigenvectors.shape[1]))
        eigenvectors = np.concatenate([eigenvectors, pad], axis=1)

    # Random sign flip per eigenvector for invariance
    sign_flip = 2 * (np.random.randint(0, 2, size=(1, k)) - 0.5)
    eigenvectors = eigenvectors * sign_flip

    return torch.tensor(eigenvectors, dtype=torch.float32, device=device)


class GNNSCM(nn.Module):
    """Generates synthetic graph node-classification datasets using a GNN-based
    Structural Causal Model (SCM).

    Initial cause variables are sampled per node, optionally augmented with
    structural features (Laplacian PE, degree, PageRank), then propagated
    through a sequence of ``SemiGraphConv`` layers (each combining a linear
    transformation with a graph convolution) interleaved with activations and
    noise. Node features ``X`` and targets ``y`` are extracted from the
    intermediate or final layer outputs, depending on the causal configuration.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph whose structure is used for message passing in the
        ``SemiGraphConv`` layers and for computing structural features.

    graph_sampler_type : str
        Graph-sampler type name used upstream to generate ``graph``.
        Not used directly here; stored for provenance.

    conv_type : str, default="GraphSAGE"
        The graph convolution variant used inside ``SemiGraphConv`` layers.
        Supported values: ``"gcn"``, ``"sage-mean"``, ``"sage-min"``,
        ``"sage-max"``, ``"gt"`` (graph transformer).

    graph_conv_ratio : float, default=1.0
        Per-dimension probability that a ``SemiGraphConv`` output dimension
        uses the graph-convolution branch rather than the plain linear branch.
        A value of 1.0 means all dimensions use graph convolution.

    avg_degree : float
        Average degree of the input graph. Not used directly here; stored for
        provenance / logging.

    n_lappe_features : int, default=0
        Number of Laplacian positional-encoding features to concatenate to the
        initial cause variables. Set to 0 to disable.

    degree_features : bool, default=False
        Whether to concatenate log-transformed in-degree as an additional input
        feature per node.

    pagerank_features : bool, default=False
        Whether to concatenate log-transformed PageRank scores as an additional
        input feature per node.

    transform_strucfeat_config : dict or None, default=None
        Optional configuration for applying a ``QuantileTransformer`` to
        structural features before concatenation. Expected keys:
        ``"output_distribution"`` (str) and ``"scaling_factor"`` (float).

    seq_len : int, default=1024
        Number of nodes (samples) to generate data for.

    num_features : int, default=100
        Number of node features in the output ``X``.

    num_outputs : int, default=1
        Dimensionality of the target ``y`` per node.

    is_causal : bool, default=True
        - If ``True``, ``X`` and ``y`` are sampled from the concatenated
          intermediate hidden states of the SCM layers. The ``num_causes``
          parameter controls the number of initial root variables.
        - If ``False``, the initial causes are used directly as ``X``, and the
          final layer output becomes ``y``. ``num_causes`` is set equal to
          ``num_features``.

    num_causes : int, default=10
        Number of initial root cause variables sampled by ``XSampler``.
        Only relevant when ``is_causal=True``.

    y_is_effect : bool, default=True
        How ``y`` is selected when ``is_causal=True``.
        - If ``True``, ``y`` is taken from the final layer outputs (terminal
          effects in the causal chain).
        - If ``False``, ``y`` is sampled from earlier intermediate outputs.

    in_clique : bool, default=False
        When ``is_causal=True``, controls whether ``X`` and ``y`` are drawn
        from a contiguous block of intermediate outputs (denser dependencies)
        or randomly and independently.

    sort_features : bool, default=True
        Whether to sort the selected feature indices from intermediate outputs.
        Only relevant when ``is_causal=True``.

    num_layers : int, default=10
        Total number of layers in the SCM network. Must be >= 2. Includes the
        initial linear layer and subsequent (Activation -> SemiGraphConv ->
        Noise) blocks.

    hidden_dim : int, default=20
        Dimensionality of hidden representations. Automatically increased when
        ``is_causal=True`` to at least ``num_outputs + 2 * num_features``.

    mlp_activations : default=nn.Tanh
        Activation function class applied before each ``SemiGraphConv`` layer.

    init_std : float, default=1.0
        Standard deviation for weight initialization.

    block_wise_dropout : bool, default=True
        Weight initialization strategy.
        - If ``True``, only random blocks within the weight matrix are
          initialized (the rest are zero), encouraging sparsity.
        - If ``False``, standard normal initialization is used, followed by a
          Bernoulli dropout mask controlled by ``mlp_dropout_prob``.

    mlp_dropout_prob : float, default=0.1
        Dropout probability for standard initialization (when
        ``block_wise_dropout=False``). Clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale ``init_std`` to compensate for variance reduction
        caused by dropout during initialization.

    sampling : str, default="normal"
        Distribution used by ``XSampler`` for the initial cause variables.
        Options: ``"normal"``, ``"uniform"``, ``"mixed"`` (combination of
        normal, multinomial, Zipf, and uniform).

    pre_sample_cause_stats : bool, default=False
        If ``True`` and ``sampling="normal"``, the per-cause mean and std are
        pre-sampled. Passed to ``XSampler``.

    noise_std : float, default=0.01
        Base standard deviation for Gaussian noise added after each
        ``SemiGraphConv`` layer (except the first plain linear layer).

    pre_sample_noise_std : bool, default=False
        If ``True``, per-dimension noise standard deviations are sampled from
        ``|N(0, noise_std)|`` rather than using a fixed value.

    device : str, default="cpu"
        Device for tensor allocation.

    **kwargs : dict
        Unused hyperparameters from parent configurations (ignored).
    """

    def __init__(
        self,
        *,
        graph: Data,
        graph_sampler_type: str,
        conv_type: str = "GraphSAGE",
        graph_conv_ratio: float = 1.0,
        avg_degree: float,
        n_lappe_features: int = 0,
        degree_features: bool = False,
        pagerank_features: bool = False,
        transform_strucfeat_config: dict | None = None,  # TODO: maybe rename?
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

        self.graph = graph

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

        self.conv_type = conv_type
        self.graph_conv_ratio = graph_conv_ratio
        self.n_lappe_features = n_lappe_features
        self.degree_features = degree_features
        self.pagerank_features = pagerank_features
        self.transform_strucfeat_config = transform_strucfeat_config

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

        # Build layers
        num_inputs = (
            self.num_causes
            + self.n_lappe_features
            + int(degree_features)
            + int(pagerank_features)
        )
        layers: list[nn.Module] = [nn.Linear(num_inputs, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

        self.initialize_parameters()

    def generate_layer_modules(self, is_output_layer=False):
        """Builds one SCM block: Activation -> SemiGraphConv -> GaussianNoise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        linear_layer = SemiGraphConv(  # TODO: rename
            d_input=self.hidden_dim,
            d_output=out_dim,
            conv_type=self.conv_type,  # type: ignore
            graph_conv_ratio=self.graph_conv_ratio,
        )

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        """Initializes parameters using block-wise dropout or normal initialization."""
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self.initialize_with_block_dropout(param, i)
            else:
                self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        """Initializes parameters using block-wise dropout."""
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

    def initialize_normally(self, param, index):
        """Initializes parameters using normal distribution."""
        if param.dim() == 2:  # Applies only to weights, not biases
            dropout_prob = (
                self.mlp_dropout_prob if index > 0 else 0
            )  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / (
                (1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1
            )
            nn.init.normal_(param, std=std)
            param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    def transform_strucfeat(self, feat: Tensor) -> Tensor:
        if self.transform_strucfeat_config is None:
            return feat
        output_distribution = self.transform_strucfeat_config["output_distribution"]
        scaling_factor = self.transform_strucfeat_config["scaling_factor"]
        transformer = QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=min(feat.shape[0], 1000),
        )
        feat = torch.tensor(transformer.fit_transform(feat.cpu().numpy()))
        feat = scaling_factor * feat
        return feat

    def compute_lappe_features(self, graph: Data) -> Tensor:
        device = graph.edge_index.device
        try:
            lappe = _compute_laplacian_pe(
                graph.edge_index, graph.num_nodes, k=self.n_lappe_features, device=device
            )
        except ArpackNoConvergence:
            lappe = torch.randn(
                [graph.num_nodes, self.n_lappe_features], device=device
            )
            print("LapPE did not converge. Replacing LapPE with random values.")
        lappe = self.transform_strucfeat(lappe)
        return lappe

    def compute_degree_features(self, graph: Data) -> Tensor:
        degrees = degree(graph.edge_index[1], num_nodes=graph.num_nodes).unsqueeze(-1)
        degrees = torch.log(1 + degrees)
        degrees = self.transform_strucfeat(degrees)
        return degrees

    def compute_pagerank_features(self, graph: Data) -> Tensor:
        alpha = 0.6 + 0.3 * np.random.rand()
        pagerank = compute_pagerank(
            graph.edge_index, graph.num_nodes, alpha=alpha, log=True, max_iterations=30
        )
        pagerank = self.transform_strucfeat(pagerank)
        return pagerank

    def forward(self):
        """Generates synthetic node features and targets by propagating sampled
        causes (optionally augmented with structural features) through the
        GNN-based SCM layers."""
        graph = self.graph
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index
        causes = self.xsampler(num_nodes).sample()  # (seq_len, num_causes)

        # Drop auxiliary node-level keys to prevent data leakage
        for key in list(graph.keys()):
            if key not in ("edge_index", "num_nodes"):
                try:
                    delattr(graph, key)
                except AttributeError:
                    pass

        inputs = [causes]
        if self.n_lappe_features > 0:
            inputs.append(self.compute_lappe_features(graph))
        if self.degree_features:
            inputs.append(self.compute_degree_features(graph))
        if self.pagerank_features:
            inputs.append(self.compute_pagerank_features(graph))
        inputs = torch.cat(inputs, dim=1)  # type: ignore

        outputs = self.propagate(inputs, edge_index, num_nodes)
        outputs = outputs[
            2:
        ]  # Start from 2 because the first layer is only linear without activation

        # Handle outputs based on causality
        X, y = self.handle_outputs(causes, outputs)

        # Check for NaNs and handle them by setting to default values
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X.cpu(), y.cpu()

    def propagate(
        self, inputs: Tensor, edge_index: Tensor, num_nodes: int
    ) -> list[Tensor]:
        """Push ``inputs`` through the SCM stack, returning every block's output.

        Split out of :meth:`forward` so a caller can supply its own inputs
        and reuse the same blocks and initialisation.
        """
        outputs = [inputs]
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, SemiGraphConv):
                    module.edge_index = edge_index
                    module.num_nodes = num_nodes
            outputs.append(layer(outputs[-1]))
        return outputs

    def handle_outputs(self, causes, outputs):
        """Selects node features ``X`` and targets ``y`` from layer outputs.

        In causal mode, ``X`` and ``y`` are sampled from the concatenated
        intermediate layer outputs. In non-causal mode, the original causes
        are used as ``X`` and the final layer output as ``y``.

        Parameters
        ----------
        causes : torch.Tensor
            Initial cause variables, shape ``(num_nodes, num_causes)``.

        outputs : list[torch.Tensor]
            Per-layer output tensors from the SCM network.

        Returns
        -------
        X : torch.Tensor
            Node features, shape ``(num_nodes, num_features)``.

        y : torch.Tensor
            Node targets, shape ``(num_nodes, num_outputs)``.
        """
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
