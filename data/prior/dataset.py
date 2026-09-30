"""Synthetic graph prior used for Wander pretraining.

Each ``SCMPrior.get_batch`` call samples one graph with node features and labels
(or a link-prediction graph). Features and labels come from structural causal
models (GNN / tree SCMs), either graph-first or features-first; ``native_backend``
can instead draw from the GraphPFN or NodePFN priors.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from .gnn_scm import GNNSCM
from .graph import (
    BipartiteLatentFactorSampler,
    DataConditionedGraphSampler,
    GraphSampler,
    sample_erdos_renyi_graph,
)
from .hp_sampling import HpSamplerList
from .mlp_scm import MLPSCM
from .prior_config import DEFAULT_SAMPLED_HP
from .reg2cls import Reg2Cls
from .tree_scm import TreeSCM

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


class RestartBatchGeneration(Exception):
    """Raised when inner dataset retries are exhausted; ``get_batch`` catches this
    and restarts from scratch (resample fixed HPs, new graph, new meta-HPs).
    """


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch
    """

    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size

    @staticmethod
    def train_size_from_frac(seq_len: int, frac: float) -> int:
        """Train split length from ``train_size_frac`` in ``DEFAULT_FIXED_HP`` (uniform in [min,max])."""
        if seq_len <= 1:
            return 0
        raw = int(round(float(seq_len) * float(frac)))
        if raw < 1:
            return 1
        if raw >= seq_len:
            return seq_len - 1
        return raw

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor) -> Tuple[Tensor, int]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant
        features.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (T, H).

        Returns
        -------
        tuple
            (X_filtered, d) where:
            - X_filtered has constant columns removed, shape (T, d)
            - d is the number of remaining features
        """
        unique_mask = torch.tensor(
            [len(torch.unique(X[:, j])) > 1 for j in range(X.shape[1])],
            dtype=torch.bool,
        )
        return X[:, unique_mask], int(unique_mask.sum().item())

    @staticmethod
    def _permute_graph(graph: Data, perm: Tensor) -> Data:
        """Reorder graph nodes according to *perm*.

        ``perm`` follows the same convention as ``y_perm = y[perm]``:
        new position *j* receives the data of old node ``perm[j]``.

        Parameters
        ----------
        graph : Data
            Original graph with *N* nodes.
        perm : Tensor
            1-D LongTensor of length *N* mapping new->old indices.

        Returns
        -------
        Data
            New graph with edges remapped.
        """
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)

        src, dst = graph.edge_index
        new_edge_index = torch.stack([inv_perm[src], inv_perm[dst]])

        new_graph = Data(
            edge_index=new_edge_index,
            num_nodes=graph.num_nodes,
        )
        return new_graph

    @staticmethod
    def sanity_check(
        X: Tensor,
        y: Tensor,
        train_size: int,
        is_regression: bool,
        n_attempts: int = 0,
        min_classes: int = 2,
        graphs: Optional[list] = None,
    ) -> bool:
        """
        Verifies that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=0
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        graphs : list of Data, optional
            One graph per batch element.  When provided and a permutation
            fixes an invalid split, the corresponding graph is also permuted
            in-place (the list entry is replaced).

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """
        if np.isnan(X).any() or np.isnan(y).any():
            return False

        def is_valid_split(xi: Tensor, yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])

            if is_regression:
                return (len(unique_tr) > 10) and (len(unique_te) > 10)

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            return (
                set(unique_tr.tolist()) == set(unique_te.tolist())
                and len(unique_tr) >= min_classes
            )

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(xi, yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                if is_valid_split(xi_perm, yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    if graphs is not None:
                        graphs[i] = Prior._permute_graph(graphs[i], perm)
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """
    Generates a single synthetic dataset using Structural Causal Models (SCM).

    Each call to ``get_batch`` samples one graph, one set of hyper-parameters,
    and produces one (X, y) pair.

    Graph size, feature count, class count, and train split fraction are configured
    via ``fixed_hp`` (see ``DEFAULT_FIXED_HP`` in ``prior_config``), not constructor
    arguments.

    If ``fixed_hp["native_backend"]`` is ``"graphpfn"`` or ``"nodepfn"``, both
    node classification and link prediction use that project's official prior
    graphs (see ``native_backends.py``). LP reuses those graphs as undirected
    label-stripped samples (no bipartite SCM fallback).

    Parameters
    ----------
    classification_only : bool, default=True
        If True, only classification tasks are generated.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Structural configuration including ``num_nodes``, ``num_features``,
        ``num_classes``, ``train_size_frac``, and ``prior_type``. Set
        ``native_backend`` to ``"graphpfn"`` or ``"nodepfn"`` to bypass SCM
        for node classification.

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation (meta-distributions).

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')

    profile_generation : bool, default=False
        If True, populate ``last_generation_profile`` after each ``get_batch``
        with total wall time and the full resolved fixed HPs dict.
    """

    def __init__(
        self,
        *,
        classification_only: bool = True,
        fixed_hp: Dict[str, Any],
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        device: str = "cpu",
        profile_generation: bool = False,
    ):
        super().__init__(batch_size=1)

        self.classification_only = classification_only
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.device = device
        self.profile_generation = profile_generation
        self.last_generation_profile: Optional[Dict[str, Any]] = None
        self._native_sampler: Any = None

    @staticmethod
    def _materialize_graph_hps(hp: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Turn sampled float HPs into integers for graph generation."""
        scm_floor = int(hp.get("scm_min_features", 4))
        seq_len = max(2, int(round(float(hp["num_nodes"]))))
        num_features = max(scm_floor, int(round(float(hp["num_features"]))))
        num_classes = max(2, int(round(float(hp["num_classes"]))))
        return seq_len, num_features, num_classes, scm_floor

    @staticmethod
    def _resolve_fixed_hp(fixed_hp: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Resolve fixed HPs: sample distribution specs, keep literal values.

        Entries that are plain values (str, bool, int, float, list, tuple)
        are returned as-is.  Entries that are dicts with a ``"distribution"``
        key are treated as single-draw distribution specs and sampled once
        via :class:`HpSamplerList`.
        """
        dist_specs = {
            k: v
            for k, v in fixed_hp.items()
            if isinstance(v, dict) and "distribution" in v
        }
        literals = {k: v for k, v in fixed_hp.items() if k not in dist_specs}

        # Reject known pathological combinations by resampling the fixed HP draw.
        # This avoids very expensive graph construction settings.
        max_resample_attempts = 1024
        for _ in range(max_resample_attempts):
            if dist_specs:
                sampled = HpSamplerList(dist_specs, device=device).sample()
                sampled = {k: v() if callable(v) else v for k, v in sampled.items()}
            else:
                sampled = {}

            resolved = {**literals, **sampled}
            features_first = bool(resolved.get("features_first", False))
            graph_sampler_type = str(resolved.get("graph_sampler_type", ""))
            num_nodes = int(round(float(resolved.get("num_nodes", 0))))
            avg_degree = float(resolved.get("avg_degree", 0.0))
            num_features_resolved: Optional[int] = None
            if "num_features" in resolved and resolved["num_features"] is not None:
                num_features_resolved = int(round(float(resolved["num_features"])))

            invalid_ml_sbm_pa = (
                (not features_first)
                and graph_sampler_type == "multi-level-sbm-with-pa"
                and (num_nodes > 10_000 or num_nodes < 1000)
            )
            invalid_degree_budget = num_nodes < avg_degree * 2
            # Peak memory ~ scales with num_nodes * num_features (dense X / model state).
            invalid_node_feature_budget = (
                num_features_resolved is not None
                and num_nodes * num_features_resolved >= 4_000_000 #8_000_000
            )
            if not (
                invalid_ml_sbm_pa
                or invalid_degree_budget
                or invalid_node_feature_budget
            ):
                return resolved

        raise RuntimeError(
            "Failed to sample valid fixed HPs after many attempts: "
            "repeatedly hit features_first=False with graph_sampler_type="
            "'multi-level-sbm-with-pa' and num_nodes outside [500, 10000], "
            "or num_nodes < 2 * avg_degree, "
            "or num_nodes * num_features >= 8_000_000."
        )

    @torch.no_grad()
    def generate_dataset(
        self,
        params: Dict[str, Any],
        n_shuffle_attempts: int = 10,
    ) -> Tuple[Tensor, Tensor, Tensor, Data | None]:
        """
        Generates a single valid dataset based on the provided parameters.

        When the initial train/test split is invalid (e.g. not all classes
        appear in both splits), the method first tries to fix it by randomly
        permuting the node ordering -- shuffling X, y **and** the graph
        together.  Only when all ``n_shuffle_attempts`` fail does it
        regenerate the dataset from scratch.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, cat_plan, etc.

        n_shuffle_attempts : int, default=10
            Number of random permutations to try before regenerating.

        Returns
        -------
        tuple
            (X, y, d, graph) where:
            - X: Features tensor of shape ``(num_nodes, d)``
            - y: Labels tensor of shape ``(num_nodes,)``
            - d: Number of active features after filtering (scalar Tensor)
            - graph: The (possibly permuted) PyG Data graph
        """

        if params["prior_type"] == "gnn_scm":
            prior_cls = GNNSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        elif params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        has_graph = params.get("graph") is not None
        n_attempts = 1
        while True:
            X, y = prior_cls(**params)()

            frac = float(params.get("train_size_frac", 0.25))
            params["train_size"] = Prior.train_size_from_frac(X.shape[0], frac)
            X, y = Reg2Cls(params)(X, y)

            X, d = self.delete_unique_features(X)

            graphs = [params["graph"]] if has_graph else [None]
            ok = self.sanity_check(
                X.unsqueeze(0),
                y.unsqueeze(0),
                params["train_size"],
                is_regression=(params["num_classes"] == 0),
                n_attempts=n_shuffle_attempts,
                graphs=graphs if has_graph else None,
            )

            if ok:
                if n_attempts > 1:
                    print(f"Needed {n_attempts} attempts to generate a valid dataset")
                return (
                    X,
                    y,
                    torch.tensor(d, device=self.device, dtype=torch.long),
                    graphs[0] if has_graph else None,
                )
            elif n_attempts > 10:
                print(
                    "Giving up on generating a valid dataset — restarting get_batch "
                    "(resample fixed HPs, graph, and meta-HPs)"
                )
                raise RestartBatchGeneration()
            else:
                n_attempts += 1

    @staticmethod
    def _build_cat_plan(
        num_features: int,
        min_features: int,
        cat_proportion_fn,
        cat_num_categories_fn,
        cat_ordered_prob_fn,
        p_cat_fn=1.0,
    ) -> Tuple[list, int]:
        """Pre-compute which GNNSCM columns become categorical and how.

        Returns the categorical plan and the number of features the SCM
        should produce (accounting for one-hot expansion).
        """
        p_cat = p_cat_fn() if callable(p_cat_fn) else p_cat_fn
        if np.random.random() >= float(p_cat):
            return [], num_features

        cat_proportion = cat_proportion_fn() if callable(cat_proportion_fn) else cat_proportion_fn
        cat_ordered_prob = cat_ordered_prob_fn() if callable(cat_ordered_prob_fn) else cat_ordered_prob_fn
        n_cat_target = round(cat_proportion * num_features)

        cat_plan: list[dict] = []
        total_expansion = 0
        for _ in range(n_cat_target):
            nc = cat_num_categories_fn() if callable(cat_num_categories_fn) else cat_num_categories_fn
            nc = max(int(nc), 2)
            is_onehot = np.random.random() > cat_ordered_prob
            expansion = (nc - 1) if is_onehot else 0

            scm_features = num_features - total_expansion - expansion
            if scm_features < min_features or len(cat_plan) + 1 > scm_features:
                break

            total_expansion += expansion
            cat_plan.append({"num_cats": nc, "is_onehot": is_onehot})

        num_features_scm = num_features - total_expansion
        return cat_plan, num_features_scm

    @torch.no_grad()
    def _get_batch_lp(
        self,
        hp: Dict[str, Any],
        seq_len: int,
        num_features: int,
        num_classes: int,
        scm_floor: int,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Generate a single synthetic link-prediction dataset.

        With probability ``bipartite_lp_prob`` builds a directed bipartite graph
        via :class:`BipartiteLatentFactorSampler`; otherwise reuses the usual
        graph+feature pipeline (graph-first or features-first) and strips labels,
        treating the graph as undirected LP.

        Independently of that choice, with probability ``lp_drop_features_prob``
        node features are dropped (``X`` is ``None`` and ``lp_meta["drop_features"]``
        is True).

        Returns ``(X, lp_meta)`` where ``lp_meta`` carries the canonical
        ``edge_index`` plus ``is_bipartite``, ``candidate_offset``, ``num_nodes``,
        ``lp_train_size_frac`` (fraction of edges kept as context/train), and
        ``drop_features``.
        """
        frac = float(hp.get("lp_train_size_frac", 0.8))
        drop_features = np.random.random() < float(hp.get("lp_drop_features_prob", 0.5))

        if np.random.random() < float(hp.get("bipartite_lp_prob", 0.5)):
            edge_index, X, num_users, num_nodes = BipartiteLatentFactorSampler(
                n_nodes=seq_len,
                num_features=num_features,
                avg_degree=float(hp["avg_degree"]),
                user_item_ratio=float(hp.get("user_item_ratio", 1.0)),
                latent_dim_k=float(hp.get("latent_dim_k", 16.0)),
                popularity_zipf_gamma=float(hp.get("popularity_zipf_gamma", 2.0)),
                feature_informativeness=float(hp.get("feature_informativeness", 1.0)),
                device=self.device,
            ).sample()
            lp_meta = {
                "edge_index": edge_index,
                "is_bipartite": True,
                "candidate_offset": int(num_users),
                "num_nodes": int(num_nodes),
                "lp_train_size_frac": frac,
            }
        else:
            features_first = hp.get("features_first", False)
            if not features_first:
                graph, X, _y, _d, seq_len, _train_size = self._get_batch_graph_first(
                    hp, seq_len, num_features, num_classes, scm_floor
                )
            else:
                graph, X, _y, _d, seq_len, _train_size = self._get_batch_features_first(
                    hp, seq_len, num_features, num_classes, scm_floor
                )
            lp_meta = {
                "edge_index": graph.edge_index,
                "is_bipartite": False,
                "candidate_offset": 0,
                "num_nodes": int(graph.num_nodes),
                "lp_train_size_frac": frac,
            }

        lp_meta["drop_features"] = bool(drop_features)
        if drop_features:
            X = None
        return X, lp_meta

    @staticmethod
    def _native_backend_name(hp: Dict[str, Any]) -> Optional[str]:
        backend = hp.get("native_backend")
        if backend is None:
            return None
        name = str(backend).strip().lower()
        if name in ("", "none", "scm"):
            return None
        return name

    def _get_batch_native_lp(self, backend: str) -> Tuple[Tensor | None, Dict[str, Any]]:
        """Reuse a native prior graph as undirected label-stripped LP."""
        graph, X, _y, _d, _seq_len, _train_size = self._get_batch_native(backend)
        hp = self.fixed_hp
        frac = float(hp.get("lp_train_size_frac", np.random.uniform(0.7, 0.9)))
        drop_features = np.random.random() < float(hp.get("lp_drop_features_prob", 0.5))
        lp_meta = {
            "edge_index": graph.edge_index,
            "is_bipartite": False,
            "candidate_offset": 0,
            "num_nodes": int(graph.num_nodes),
            "lp_train_size_frac": frac,
            "drop_features": bool(drop_features),
        }
        if drop_features:
            X = None
        return X, lp_meta

    def _get_batch_native(self, backend: str) -> Tuple[Data, Tensor, Tensor, Tensor, int, int]:
        """Sample one node_cls graph from the official GraphPFN / NodePFN prior."""
        if self._native_sampler is None:
            from .native_backends import GraphPFNNativeSampler, NodePFNNativeSampler

            if backend == "graphpfn":
                self._native_sampler = GraphPFNNativeSampler(
                    device=self.device,
                    classification_only=self.classification_only,
                    n_workers=0,
                )
            elif backend == "nodepfn":
                self._native_sampler = NodePFNNativeSampler(device=self.device)
            else:
                raise ValueError(
                    f"Unknown native_backend {backend!r}; expected 'graphpfn' or 'nodepfn'"
                )
        return self._native_sampler.sample()

    @torch.no_grad()
    def get_batch(
        self,
        *,
        task_type: str = "node_cls",
        profile_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Data | None, Tensor, Tensor, Tensor, int, int] | Tuple[Tensor, Dict[str, Any]]:
        """
        Generate a single synthetic dataset.

        Parameters
        ----------
        task_type : {"node_cls", "lp"}
            ``"node_cls"`` (default) returns the 6-tuple described below.
            When ``fixed_hp["native_backend"]`` is ``"graphpfn"`` or ``"nodepfn"``,
            both ``node_cls`` and ``lp`` use that project's official prior graphs.
            ``"lp"`` reuses those graphs as undirected label-stripped LP (see
            :meth:`_get_batch_native_lp`). SCM ``_get_batch_lp`` is unused.

        Returns
        -------
        graph : Data or None
            The sampled graph.

        X : Tensor
            Features tensor of shape (num_nodes, d).

        y : Tensor
            Labels tensor of shape (num_nodes,).

        d : Tensor
            Number of active features after filtering (scalar tensor).

        seq_len : int
            Sequence length (number of nodes).

        train_size : int
            Position for the train/test split.
        """
        self.last_generation_profile = None
        prof: Optional[Dict[str, Any]] = None
        if self.profile_generation:
            prof = dict(profile_context or {})
            prof["event"] = "synthetic_prior_generation"

        while True:
            t_wall = time.perf_counter()
            try:
                hp = self._resolve_fixed_hp(self.fixed_hp, self.device)

                backend = self._native_backend_name(hp)
                if backend and task_type == "node_cls":
                    out = self._get_batch_native(backend)
                    if prof is not None:
                        prof["resolved_fixed_hp"] = {k: v for k, v in hp.items()}
                        prof["native_backend"] = backend
                        graph = out[0]
                        if graph is not None:
                            prof["final_num_nodes"] = int(graph.num_nodes)
                            prof["final_edge_index_size"] = int(graph.edge_index.shape[1])
                        prof["ms_wall_get_batch"] = (time.perf_counter() - t_wall) * 1000.0
                        self.last_generation_profile = prof
                    return out

                if backend and task_type == "lp":
                    out_lp = self._get_batch_native_lp(backend)
                    if prof is not None:
                        prof["resolved_fixed_hp"] = {k: v for k, v in hp.items()}
                        prof["native_backend"] = backend
                        prof["native_lp"] = True
                        prof["final_num_nodes"] = int(out_lp[1]["num_nodes"])
                        prof["ms_wall_get_batch"] = (time.perf_counter() - t_wall) * 1000.0
                        self.last_generation_profile = prof
                    return out_lp

                seq_len, num_features, num_classes, scm_floor = self._materialize_graph_hps(
                    hp
                )

                if prof is not None:
                    prof["resolved_fixed_hp"] = {k: v for k, v in hp.items()}

                if task_type == "lp":
                    return self._get_batch_lp(
                        hp, seq_len, num_features, num_classes, scm_floor
                    )

                if task_type != "node_cls":
                    raise ValueError(
                        f"task_type must be 'node_cls' or 'lp'; got {task_type!r}"
                    )

                features_first = hp.get("features_first", False)

                if not features_first:
                    out = self._get_batch_graph_first(
                        hp, seq_len, num_features, num_classes, scm_floor
                    )
                else:
                    out = self._get_batch_features_first(
                        hp, seq_len, num_features, num_classes, scm_floor
                    )

                graph, X, y, d, seq_len, train_size = out
                if graph is not None and np.random.random() < float(
                    hp.get("random_graph_prob", 0.0)
                ):
                    graph = sample_erdos_renyi_graph(
                        int(graph.num_nodes),
                        float(hp["avg_degree"]),
                        self.device,
                    )
                out = (graph, X, y, d, seq_len, train_size)

                if prof is not None:
                    graph = out[0]
                    if graph is not None:
                        prof["final_num_nodes"] = int(graph.num_nodes)
                        prof["final_edge_index_size"] = int(graph.edge_index.shape[1])
                    prof["ms_wall_get_batch"] = (time.perf_counter() - t_wall) * 1000.0
                    self.last_generation_profile = prof

                return out
            except RestartBatchGeneration:
                continue

    def _get_batch_graph_first(
        self,
        hp: Dict[str, Any],
        seq_len: int,
        num_features: int,
        num_classes: int,
        scm_floor: int,
    ) -> Tuple[Data | None, Tensor, Tensor, Tensor, int, int]:
        """Original path: sample graph first, then generate features/labels."""
        gsampler = GraphSampler(
            n_nodes=seq_len,
            avg_degree=hp["avg_degree"],
            sampler_type=hp["graph_sampler_type"],
            device=self.device,
            sampler_kwargs=hp,
        )
        while True:
            graph = gsampler.sample()
            if graph.num_nodes < seq_len // 2:
                print(
                    f"GraphSampler shrunk node count from {seq_len} to {graph.num_nodes} (graph first)"
                )
                continue
            break

        seq_len = graph.num_nodes

        meta_hp = HpSamplerList(self.sampled_hp, device=self.device).sample()
        cat_num_categories_fn = meta_hp.pop("cat_num_categories", 2)
        meta_hp = {k: v() if callable(v) else v for k, v in meta_hp.items()}

        prior_type = hp["prior_type"]

        if self.classification_only:
            num_classes_task = num_classes
        elif np.random.random() < 0.33:
            num_classes_task = 0  # regression
        elif np.random.random() < 0.5:
            num_classes_task = num_classes
        else:
            num_classes_task = 2

        x_is_onehot_y = (
            num_classes_task >= 2
            and np.random.random() < hp.get("x_onehot_y_prob", 0.0)
        )

        if x_is_onehot_y:
            cat_plan: list[dict] = []
            num_features_scm = scm_floor
        else:
            cat_plan, num_features_scm = self._build_cat_plan(
                num_features,
                scm_floor,
                hp.get("cat_proportion", 0.0),
                cat_num_categories_fn,
                hp.get("cat_ordered_prob", 0.3),
                hp.get("p_cat", 1.0),
            )

        params = {
            **hp,
            **meta_hp,
            "graph": graph,
            "seq_len": seq_len,
            "prior_type": prior_type,
            "num_features": num_features_scm,
            "num_classes": num_classes_task,
            "cat_plan": cat_plan,
            "x_is_onehot_y": x_is_onehot_y,
            "device": self.device,
        }
        if x_is_onehot_y:
            params["x_onehot_output_dim"] = num_features

        X, y, d, graph = self.generate_dataset(params)

        return graph, X, y, d, params["seq_len"], params["train_size"]

    def _get_batch_features_first(
        self,
        hp: Dict[str, Any],
        seq_len: int,
        num_features: int,
        num_classes: int,
        scm_floor: int,
    ) -> Tuple[Data | None, Tensor, Tensor, Tensor, int, int]:
        """New path: generate features/labels first, then build graph from data.

        If the data-conditioned graph sampler shrinks the node count below
        ``seq_len // 2`` twice for the same dataset, the entire generation
        (meta-HPs, dataset, graph) is restarted to avoid an infinite loop
        when the sampler is deterministic.
        """
        while True:  # outer restart loop
            meta_hp = HpSamplerList(self.sampled_hp, device=self.device).sample()
            cat_num_categories_fn = meta_hp.pop("cat_num_categories", 2)
            meta_hp = {k: v() if callable(v) else v for k, v in meta_hp.items()}

            prior_type = hp["prior_type"]
            if prior_type == "gnn_scm":
                prior_type = "mlp_scm"

            if self.classification_only:
                num_classes_task = num_classes
            elif np.random.random() < 0.33:
                num_classes_task = 0  # regression
            elif np.random.random() < 0.5:
                num_classes_task = num_classes
            else:
                num_classes_task = 2

            x_is_onehot_y = (
                num_classes_task >= 2
                and np.random.random() < hp.get("x_onehot_y_prob", 0.0)
            )

            if x_is_onehot_y:
                cat_plan: list[dict] = []
                num_features_scm = scm_floor
            else:
                cat_plan, num_features_scm = self._build_cat_plan(
                    num_features,
                    scm_floor,
                    hp.get("cat_proportion", 0.0),
                    cat_num_categories_fn,
                    hp.get("cat_ordered_prob", 0.3),
                    hp.get("p_cat", 1.0),
                )

            params = {
                **hp,
                **meta_hp,
                "seq_len": seq_len,
                "prior_type": prior_type,
                "num_features": num_features_scm,
                "num_classes": num_classes_task,
                "cat_plan": cat_plan,
                "x_is_onehot_y": x_is_onehot_y,
                "device": self.device,
            }
            if x_is_onehot_y:
                params["x_onehot_output_dim"] = num_features

            X, y, d, _ = self.generate_dataset(params)

            graph_shrink_count = 0
            restart = False
            while True:
                graph, component_nodes = DataConditionedGraphSampler(
                    X,
                    y,
                    avg_degree=hp["avg_degree"],
                    sampler_type=hp.get("data_graph_sampler_type", "label-sbm"),
                    homophily_ratio=hp.get("data_graph_homophily_ratio", 0.5),
                    sbm_noise=hp.get("data_graph_sbm_noise", 0.0),
                    edge_method=hp.get("data_graph_edge_method", "threshold"),
                    kernel_edge_perturb_prob=hp.get(
                        "data_graph_kernel_edge_perturb_prob", 0.0
                    ),
                    ws_beta_min=hp.get("data_graph_ws_beta_min", 0.01),
                    ws_beta_max=hp.get("data_graph_ws_beta_max", 0.3),
                    ws_label_order_flip_fraction=hp.get(
                        "data_graph_ws_label_order_flip_fraction", 0.0
                    ),
                    device=self.device,
                ).sample()
                if graph.num_nodes >= seq_len // 2:
                    break
                graph_shrink_count += 1
                print(
                    f"DataConditionedGraphSampler shrunk node count "
                    f"from {seq_len} to {graph.num_nodes} "
                    f"(attempt {graph_shrink_count})"
                    f"average degree: {hp['avg_degree']}"
                )
                if graph_shrink_count >= 2:
                    restart = True
                    break

            if restart:
                print("Restarting features-first generation from scratch")
                continue

            if graph.num_nodes < X.shape[0]:
                X = X[component_nodes]
                y = y[component_nodes]

            seq_len = graph.num_nodes
            train_size = Prior.train_size_from_frac(
                seq_len, float(hp["train_size_frac"])
            )
            params["seq_len"] = seq_len
            params["train_size"] = train_size

            return graph, X, y, d, seq_len, train_size
