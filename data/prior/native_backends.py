"""Native GraphPFN / NodePFN samplers returning SCMPrior.get_batch tuples.

Used when ``fixed_hp["native_backend"]`` is ``"graphpfn"`` or ``"nodepfn"``.
Outputs match SCMPrior node_cls: ``(graph, X, y, d, seq_len, train_size)``.
"""

from __future__ import annotations

import copy
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from baselines.paths import GRAPHPFN_ROOT, NODEPFN_ROOT, require_checkout

_DEFAULT_GRAPHPFN_TOML = (
    GRAPHPFN_ROOT / "exp" / "graphpfn" / "pretrain" / "main" / "pretrain.toml"
)

ScmBatch = Tuple[Data, Tensor, Tensor, Tensor, int, int]


def _ensure_graphpfn_on_path() -> None:
    require_checkout("graphpfn")
    root = str(GRAPHPFN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("DGLBACKEND", "pytorch")


def _install_graphpfn_prior_import_shims() -> None:
    """Load GraphPFN prior modules without running ``lib/__init__.py``.

    That package init pulls the training stack (tomli_w, optuna, delu, ...).
    The prior itself only needs DGL + a ``TaskType`` enum.
    """
    import enum
    import types

    lib_dir = GRAPHPFN_ROOT / "lib"
    loaded = sys.modules.get("lib")
    if loaded is not None and getattr(loaded, "__file__", None):
        return

    def _ns(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        mod.__package__ = name
        sys.modules[name] = mod

    _ns("lib", lib_dir)
    _ns("lib.graphpfn", lib_dir / "graphpfn")
    _ns("lib.graphpfn.prior", lib_dir / "graphpfn" / "prior")

    if "lib.util" not in sys.modules:
        util = types.ModuleType("lib.util")

        class TaskType(enum.Enum):
            REGRESSION = "regression"
            BINCLASS = "binclass"
            MULTICLASS = "multiclass"

        util.TaskType = TaskType
        sys.modules["lib.util"] = util


def _strip_regression_task_choices(obj: Any) -> None:
    """Remove ``regression`` from GraphPFN task-type choice distributions."""
    if isinstance(obj, dict):
        if obj.get("_distribution_") == "choice" and isinstance(obj.get("values"), list):
            values = obj["values"]
            if values and all(isinstance(v, str) for v in values):
                kept = [v for v in values if str(v).lower() != "regression"]
                if kept:
                    obj["values"] = kept
        for value in obj.values():
            _strip_regression_task_choices(value)
    elif isinstance(obj, list):
        for value in obj:
            _strip_regression_task_choices(value)


def _require_graphpfn_prior_deps() -> None:
    try:
        import dgl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "prior_config=graphpfn samples GraphPFN's official prior, which "
            "builds DGL graphs. Install DGL in this environment "
            "(install a DGL wheel that matches this env's PyTorch/CUDA)."
        ) from exc


def _ensure_nodepfn_on_path() -> None:
    require_checkout("nodepfn")
    root = str(NODEPFN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _is_regression_task(task_type: Any) -> bool:
    name = getattr(task_type, "value", task_type)
    return str(name).lower() == "regression"


def graphpfn_dataset_to_scm_batch(dataset: dict[str, Any], device: str = "cpu") -> ScmBatch:
    """Map a GraphPFN ``PriorDataset`` dict to an SCMPrior node_cls batch."""
    features = dataset["features"].float().to(device)
    labels = dataset["labels"].to(device)
    edges = dataset["edges"].long().to(device)
    train_size = int(dataset["n_train_nodes"])
    num_nodes = int(features.shape[0])

    edge_index = to_undirected(edges, num_nodes=num_nodes)
    graph = Data(edge_index=edge_index, num_nodes=num_nodes)

    y = labels
    y_long = y.long()
    if _is_regression_task(dataset["task_type"]) or (y.float() != y_long.float()).any():
        # Should be skipped when classification_only=True; keep a safe fallback.
        n_cls = max(2, min(10, int(y.unique().numel() ** 0.5)))
        quantiles = torch.linspace(0, 1, n_cls + 1, device=y.device)[1:-1]
        thresholds = torch.quantile(y.float(), quantiles)
        y_long = torch.bucketize(y.float(), thresholds)

    d = torch.tensor(int(features.shape[1]), device=device)
    return graph, features, y_long, d, num_nodes, train_size


class GraphPFNNativeSampler:
    """Sample from GraphPFN's official prior (``sample_configs`` + ``sample_dataset``)."""

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        device: str = "cpu",
        classification_only: bool = True,
        seed: int = 0,
        n_workers: int = 0,
    ) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_GRAPHPFN_TOML
        self.device = device
        self.classification_only = classification_only
        self.seed = seed
        self.n_workers = n_workers
        self._prior_config: Optional[dict] = None

    def _ensure_sampler(self) -> None:
        if self._prior_config is not None:
            return
        _ensure_graphpfn_on_path()
        _install_graphpfn_prior_import_shims()
        _require_graphpfn_prior_deps()
        with self.config_path.open("rb") as f:
            pretrain_config = tomllib.load(f)
        prior_config = copy.deepcopy(pretrain_config["base_config"]["prior"])
        if self.classification_only:
            _strip_regression_task_choices(prior_config)
        self._prior_config = prior_config

    def sample(self) -> ScmBatch:
        self._ensure_sampler()
        from lib.graphpfn.prior.checks import SanityCheckError, check_dataset  # noqa: WPS433
        from lib.graphpfn.prior.config import sample_configs  # noqa: WPS433
        from lib.graphpfn.prior.priors import sample_dataset  # noqa: WPS433

        assert self._prior_config is not None
        max_tries = 64
        last_err: Optional[BaseException] = None
        for _ in range(max_tries):
            try:
                config = sample_configs(self._prior_config, 1)[0]
                dataset = None
                for _attempt in range(3):
                    try:
                        dataset = sample_dataset(config["prior"])
                        task_config = config["prior"]["task"]
                        n_classes = (
                            task_config["n_classes"]
                            if task_config["_type_"] == "multiclass"
                            else None
                        )
                        check_dataset(
                            features=dataset["features"],
                            labels=dataset["labels"],
                            n_train_nodes=dataset["n_train_nodes"],
                            task_type=dataset["task_type"],
                            min_features=config["sanity_check"]["min_features"],
                            n_classes=n_classes,
                        )
                        break
                    except SanityCheckError as exc:
                        last_err = exc
                        dataset = None
                if dataset is None:
                    continue
                if self.classification_only and _is_regression_task(dataset["task_type"]):
                    continue
                return graphpfn_dataset_to_scm_batch(dataset, device=self.device)
            except Exception as exc:  # noqa: BLE001 — resample like SCM RestartBatchGeneration
                last_err = exc
                continue
        raise RuntimeError(
            "GraphPFNNativeSampler: failed to draw a classification sample "
            f"after {max_tries} attempts"
        ) from last_err


def _build_nodepfn_pretrain_config() -> dict:
    """Mirror ``NodePFN/nodepfn/pretrain.py`` reload_config + __main__ overrides."""
    _ensure_nodepfn_on_path()
    try:
        from priors.utils import uniform_int_sampler_f  # noqa: WPS433
        from scripts.model_configs import (  # noqa: WPS433
            evaluate_hypers,
            get_prior_config,
        )
    except ImportError as exc:
        raise ImportError(
            "prior_config=nodepfn samples NodePFN's official prior stack "
            "(gpytorch, ConfigSpace, ...). Install those packages in this "
            "environment (the nodepfn_env job env already has them)."
        ) from exc

    max_features = 100
    config = get_prior_config(config_type="causal")
    config["prior_type"], config["differentiable"], config["flexible"] = (
        "prior_bag",
        True,
        True,
    )
    config["epochs"] = 20
    config["recompute_attn"] = True
    config["max_features"] = max_features
    config["max_num_classes"] = 20
    config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])
    config["balanced"] = False

    config["bptt_extra_samples"] = None
    config["output_multiclass_ordered_p"] = 0.0
    del config["differentiable_hyperparameters"]["output_multiclass_ordered_p"]

    config["multiclass_type"] = "rank"
    del config["differentiable_hyperparameters"]["multiclass_type"]

    config["sampling"] = "mixed"
    del config["differentiable_hyperparameters"]["sampling"]

    config["pre_sample_causes"] = True
    config["multiclass_loss_type"] = "nono"
    config["normalize_to_ranking"] = False
    config["categorical_feature_p"] = 0.2
    config["nan_prob_no_reason"] = 0.0
    config["nan_prob_unknown_reason"] = 0.0
    config["set_value_to_nan"] = 0.1
    config["new_mlp_per_example"] = True
    config["prior_mlp_scale_weights_sqrt"] = True
    config["batch_size_per_gp_sample"] = None
    config["normalize_ignore_label_too"] = True
    config["differentiable_hps_as_style"] = False
    config["max_eval_pos"] = 1000
    config["min_eval_pos"] = 100
    config["random_feature_rotation"] = True
    config["rotate_normalized_labels"] = True
    config["mix_activations"] = True
    config["emsize"] = 512
    config["nhead"] = config["emsize"] // 128
    config["bptt"] = 1024
    config["seq_len_used"] = 1024
    config["num_features"] = max_features
    config["canonical_y_encoder"] = False
    config["aggregate_k_gradients"] = 8
    config["batch_size"] = 8
    config["num_steps"] = 1024
    config["train_mixed_precision"] = True
    config["efficient_eval_masking"] = True
    config["pos_encoder"] = "none"
    config["verbose"] = False

    # Sample non-differentiable ConfigSpace HPs once (as in pretrain).
    config_sample = evaluate_hypers(config)
    # Keep differentiable distributions for per-batch sampling.
    config_sample["differentiable_hyperparameters"] = config[
        "differentiable_hyperparameters"
    ]
    config_sample["prior_type"] = "prior_bag"
    config_sample["differentiable"] = True
    config_sample["flexible"] = True
    config_sample["num_classes"] = config["num_classes"]
    config_sample["bptt"] = 1024
    config_sample["seq_len_used"] = 1024
    config_sample["max_features"] = max_features
    config_sample["num_features"] = max_features
    config_sample["max_num_classes"] = 20
    config_sample["emsize"] = 512
    return config_sample


def _make_nodepfn_get_batch(config: dict) -> tuple[Callable, dict, dict]:
    """Build the same get_batch stack as NodePFN ``get_model`` (bag+flex+diff)."""
    _ensure_nodepfn_on_path()
    import priors as priors  # noqa: WPS433
    from scripts.model_builder import (  # noqa: WPS433
        get_gp_prior_hyperparameters,
        get_mlp_prior_hyperparameters,
    )

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(
            batch_size,
            seq_len,
            num_features,
            hyperparameters,
            device,
            model_proto=model_proto,
            **kwargs,
        ):
            kwargs = {**extra_kwargs, **kwargs}
            return model_proto.get_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                hyperparameters=hyperparameters,
                num_features=num_features,
                **kwargs,
            )

        return new_get_batch

    get_batch_gp = make_get_batch(priors.fast_gp)
    get_batch_mlp = make_get_batch(priors.mlp)
    get_batch_gp = make_get_batch(priors.flexible_categorical, get_batch=get_batch_gp)
    get_batch_mlp = make_get_batch(priors.flexible_categorical, get_batch=get_batch_mlp)

    prior_bag_hyperparameters = {
        "prior_bag_get_batch": (get_batch_gp, get_batch_mlp),
        "prior_bag_exp_weights_1": 2.0,
    }
    prior_hyperparameters = {
        **get_mlp_prior_hyperparameters(config),
        **get_gp_prior_hyperparameters(config),
        **prior_bag_hyperparameters,
    }
    prior_hyperparameters["normalize_labels"] = True
    prior_hyperparameters["check_is_compatible"] = True
    prior_hyperparameters["prior_mlp_scale_weights_sqrt"] = config.get(
        "prior_mlp_scale_weights_sqrt"
    )
    prior_hyperparameters["rotate_normalized_labels"] = config.get(
        "rotate_normalized_labels", True
    )
    # Carry through flexible / network keys used by FlexibleCategorical.
    for key in (
        "num_classes",
        "balanced",
        "multiclass_type",
        "output_multiclass_ordered_p",
        "normalize_to_ranking",
        "normalize_by_used_features",
        "categorical_feature_p",
        "nan_prob_no_reason",
        "nan_prob_a_reason",
        "nan_prob_unknown_reason",
        "set_value_to_nan",
        "seq_len_used",
        "graph_type",
        "homophily_rate",
        "p_in",
        "edge_prob",
        "normalize_ignore_label_too",
        "new_mlp_per_example",
        "mix_activations",
        "verbose",
    ):
        if key in config:
            prior_hyperparameters[key] = config[key]

    get_batch_bag = make_get_batch(priors.prior_bag)
    get_batch_diff = make_get_batch(
        priors.differentiable_prior,
        get_batch=get_batch_bag,
        differentiable_hyperparameters=config["differentiable_hyperparameters"],
    )
    return get_batch_diff, prior_hyperparameters, config["differentiable_hyperparameters"]


class NodePFNNativeSampler:
    """Sample graphs from NodePFN's differentiable prior_bag + flexible categorical."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._ready = False
        self._get_batch: Optional[Callable] = None
        self._hyperparameters: Optional[dict] = None
        self._diff_hps: Optional[dict] = None
        self._bptt = 1024
        self._num_features = 100
        self._single_eval_pos_gen: Optional[Callable[[], int]] = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        config = _build_nodepfn_pretrain_config()
        get_batch, hps, diff_hps = _make_nodepfn_get_batch(config)
        self._get_batch = get_batch
        self._hyperparameters = hps
        self._diff_hps = diff_hps
        self._bptt = int(config.get("bptt", 1024))
        self._num_features = int(config.get("num_features", 100))
        from utils import get_uniform_single_eval_pos_sampler  # noqa: WPS433

        max_eval = int(config.get("max_eval_pos", self._bptt))
        min_eval = int(config.get("min_eval_pos", 100))
        self._single_eval_pos_gen = get_uniform_single_eval_pos_sampler(
            max_eval, min_len=min_eval
        )
        self._ready = True

    def sample(self) -> ScmBatch:
        self._ensure_ready()
        assert self._get_batch is not None
        assert self._hyperparameters is not None
        assert self._single_eval_pos_gen is not None

        max_tries = 16
        last_err: Optional[BaseException] = None
        for _ in range(max_tries):
            try:
                single_eval_pos = int(self._single_eval_pos_gen())
                single_eval_pos = max(1, min(single_eval_pos, self._bptt - 1))
                out = self._get_batch(
                    batch_size=1,
                    seq_len=self._bptt,
                    num_features=self._num_features,
                    hyperparameters=self._hyperparameters,
                    device=self.device,
                    single_eval_pos=single_eval_pos,
                )
                # differentiable_prior returns (x, y, y_, edge_index, style)
                if len(out) == 5:
                    x, y, _y_, edge_index, _style = out
                else:
                    x, y, _y_, edge_index = out

                # x: (T, B, H), y: (T, B)
                x0 = x[:, 0, :].detach().float().to(self.device)
                y0 = y[:, 0].detach().to(self.device)
                # Drop trailing all-zero padded feature columns when present.
                if x0.numel() and x0.shape[1] > 1:
                    used = (x0.abs().sum(dim=0) > 0).nonzero(as_tuple=False).view(-1)
                    if used.numel() > 0:
                        x0 = x0[:, : int(used.max().item()) + 1]

                y_long = y0.long()
                # Ignore NodePFN ignore_index (-100) if present.
                valid = y_long >= 0
                if not bool(valid.all()):
                    # Rare incompatible-label marker; resample.
                    if not bool(valid.any()):
                        continue
                num_nodes = int(x0.shape[0])
                edge_index = edge_index.long().to(self.device)
                edge_index = to_undirected(edge_index, num_nodes=num_nodes)
                graph = Data(edge_index=edge_index, num_nodes=num_nodes)
                d = torch.tensor(int(x0.shape[1]), device=self.device)
                train_size = max(1, min(single_eval_pos, num_nodes - 1))
                return graph, x0, y_long, d, num_nodes, train_size
            except Exception as exc:  # noqa: BLE001 — resample like SCM RestartBatchGeneration
                last_err = exc
                continue
        raise RuntimeError(
            f"NodePFNNativeSampler: failed after {max_tries} attempts"
        ) from last_err


# Module-level caches keyed by (backend, device, classification_only)
_SAMPLER_CACHE: dict[tuple[Any, ...], Any] = {}


def get_native_sampler(
    backend: str,
    *,
    device: str = "cpu",
    classification_only: bool = True,
) -> Any:
    key = (backend, device, classification_only)
    if key not in _SAMPLER_CACHE:
        if backend == "graphpfn":
            _SAMPLER_CACHE[key] = GraphPFNNativeSampler(
                device=device,
                classification_only=classification_only,
                n_workers=0,
            )
        elif backend == "nodepfn":
            _SAMPLER_CACHE[key] = NodePFNNativeSampler(device=device)
        else:
            raise ValueError(f"Unknown native_backend {backend!r}")
    return _SAMPLER_CACHE[key]


def sample_native_scm_batch(
    backend: str,
    *,
    device: str = "cpu",
    classification_only: bool = True,
) -> ScmBatch:
    return get_native_sampler(
        backend, device=device, classification_only=classification_only
    ).sample()
