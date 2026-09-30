"""Per-dataset NodePFN inference flags from upstream ``run.sh``."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NodePFNRunSpec:
    dataset: str
    n_components: int
    smoothing_steps: int
    dim_reduction: str = "tsvd"
    runs: int = 5
    n_ensemble: int | None = None
    svd_algorithm: str | None = None
    label_num_per_class: int | None = None
    cpu: bool = False
    hparams_source: str = "official"


# Official NodePFN ``run.sh`` hyper-parameters (checkpoint_epoch_30).
NODEPFN_RUN_SPECS: tuple[NodePFNRunSpec, ...] = (
    NodePFNRunSpec("cora", 15, 4),
    NodePFNRunSpec("citeseer", 15, 2),
    NodePFNRunSpec("pubmed", 15, 2),
    NodePFNRunSpec("air-usa", 25, 3, n_ensemble=8, label_num_per_class=20),
    NodePFNRunSpec("air-europe", 25, 1, n_ensemble=32, label_num_per_class=20),
    NodePFNRunSpec("air-brazil", 25, 3, n_ensemble=32, label_num_per_class=20),
    NodePFNRunSpec("wikics", 15, 2, n_ensemble=32, svd_algorithm="arpack", cpu=True),
    NodePFNRunSpec("amazon-computer", 15, 3, svd_algorithm="arpack"),
    NodePFNRunSpec("amazon-photo", 15, 3, svd_algorithm="arpack"),
    NodePFNRunSpec("dblp", 25, 3),
    NodePFNRunSpec("coauthor-cs", 25, 2, n_ensemble=32, svd_algorithm="arpack"),
    NodePFNRunSpec("coauthor-physics", 15, 4, n_ensemble=4, svd_algorithm="randomized"),
    NodePFNRunSpec("cornell", 15, 0, n_ensemble=16, svd_algorithm="randomized"),
    NodePFNRunSpec("texas", 20, 0, runs=10),
    NodePFNRunSpec("wisconsin", 25, 0, n_ensemble=32, svd_algorithm="randomized"),
    NodePFNRunSpec("chameleon", 25, 0, n_ensemble=16),
    NodePFNRunSpec("actor", 10, 0, n_ensemble=32),
    NodePFNRunSpec("amazon-ratings", 20, 3, n_ensemble=8, runs=3),
    NodePFNRunSpec("squirrel", 15, 2, svd_algorithm="arpack"),
)

# NC slug (no ``-nofeat`` suffix) -> official ``run.sh`` dataset name.
# Preprocessing (smoothing / TSVD / ensemble) is taken from that official spec.
NC_SLUG_TO_OFFICIAL: dict[str, str] = {
    "actor": "actor",
    "amazon-computers": "amazon-computer",
    "amazon-photo": "amazon-photo",
    "amazon-ratings": "amazon-ratings",
    "brazil": "air-brazil",
    "chameleon": "chameleon",
    "chameleon-filtered": "chameleon",
    "citeseer": "citeseer",
    "coauthor-cs": "coauthor-cs",
    "coauthor-physics": "coauthor-physics",
    "cora": "cora",
    "cornell": "cornell",
    "europe": "air-europe",
    "full-dblp": "dblp",
    "pubmed": "pubmed",
    "squirrel": "squirrel",
    "squirrel-filtered": "squirrel",
    "texas": "texas",
    "usa": "air-usa",
    "wiki-cs": "wikics",
    "wisconsin": "wisconsin",
}

# Upstream ``load_dataset`` applies ``T.NormalizeFeatures()`` (row L1) here.
ROW_L1_NORMALIZE_SLUGS: frozenset[str] = frozenset(
    {"actor", "coauthor-cs", "coauthor-physics"}
)

# Not in ``run.sh``: argparse defaults in ``node_classification.py``.
_NC_EXTRA_SPECS: dict[str, NodePFNRunSpec] = {
    "roman-empire": NodePFNRunSpec(
        "roman-empire",
        50,
        0,
        dim_reduction="none",
        n_ensemble=32,
        hparams_source="argparse-default",
    ),
}

# Inherent no-feature graphs (slug does not use the ``-nofeat`` suffix).
INHERENT_NOFEAT_SLUGS: frozenset[str] = frozenset({"cycles", "grids"})

# Fallback when there is no official / extra spec (GraphLand, Full Cora, …).
DEFAULT_NC_RUN_SPEC = NodePFNRunSpec(
    "wander-default",
    15,
    2,
    n_ensemble=32,
    svd_algorithm="arpack",
    hparams_source="wander-default",
)

NODEPFN_MAX_CLASSES = 20
# Structure-only: one shared constant; skip TSVD (sklearn needs ≥2 columns).
NOFEAT_STATIC_DIM = 1
CHECKPOINT_EPOCH = 30

# Paper Table 9 search (homophily); we only vary these two axes for paper NC graphs.
GRID_TSVD_COMPONENTS: tuple[int, ...] = (10, 15, 20, 25, 30)
GRID_SMOOTHING_STEPS: tuple[int, ...] = (0, 1)


def spec_by_dataset(name: str) -> NodePFNRunSpec:
    key = name.strip().lower()
    for spec in NODEPFN_RUN_SPECS:
        if spec.dataset == key:
            return spec
    raise KeyError(f"Unknown NodePFN dataset {name!r}")


def _base_slug(slug: str) -> str:
    raw = slug.strip().lower()
    if raw.endswith("-nofeat"):
        return raw[: -len("-nofeat")]
    return raw


def is_nofeat_slug(slug: str) -> bool:
    raw = slug.strip().lower()
    return raw.endswith("-nofeat") or _base_slug(raw) in INHERENT_NOFEAT_SLUGS


def nofeat_structure_only_spec(spec: NodePFNRunSpec) -> NodePFNRunSpec:
    """One constant feature, no GCN smoothing, no TSVD (sklearn needs ≥2 cols)."""
    return replace(
        spec,
        n_components=1,
        smoothing_steps=0,
        dim_reduction="none",
        hparams_source=f"{spec.hparams_source}+nofeat-const1",
    )


def ensemble_size(spec: NodePFNRunSpec) -> int:
    """Upstream argparse default is 32 when ``run.sh`` omits ``--n_ensemble``."""
    return 32 if spec.n_ensemble is None else int(spec.n_ensemble)


def svd_algorithm(spec: NodePFNRunSpec) -> str:
    return spec.svd_algorithm or "arpack"


def spec_for_nc_slug(slug: str) -> NodePFNRunSpec:
    """Map an NC slug onto NodePFN preprocessing hyper-parameters.

    Official ``run.sh`` flags are reused when the graph matches a paper dataset.
    Device (``cpu=True`` on WikiCS) is not inherited — eval uses CUDA unless
    ``--cpu`` is passed. ``label_num_per_class`` is unused (paper splits are fixed).
    """
    base = _base_slug(slug)
    official_name = NC_SLUG_TO_OFFICIAL.get(base)
    if official_name is None:
        spec = replace(DEFAULT_NC_RUN_SPEC, dataset=base)
    else:
        official = spec_by_dataset(official_name)
        spec = replace(
            official,
            cpu=False,
            label_num_per_class=None,
            n_ensemble=ensemble_size(official),
            svd_algorithm=svd_algorithm(official),
            hparams_source=f"official:{official.dataset}",
        )
    extra = _NC_EXTRA_SPECS.get(base)
    if extra is not None:
        spec = extra
    if is_nofeat_slug(slug):
        return nofeat_structure_only_spec(spec)
    return spec


def inference_runs(
    registry_key: str,
    spec: NodePFNRunSpec,
    override: int | None = None,
) -> int:
    """TSVD / ensemble seeds per data split.

    Several Wander protocol splits: 1 seed (std is split variance). Otherwise
    keep the official ``run.sh`` ``--runs`` count.
    """
    from data.nc_splits import n_eval_nc_splits

    if override is not None and int(override) > 0:
        return int(override)
    if n_eval_nc_splits(registry_key) > 1:
        return 1
    return int(spec.runs)


def uses_row_l1_normalize(slug: str) -> bool:
    return _base_slug(slug) in ROW_L1_NORMALIZE_SLUGS


def grid_cell_tag(n_components: int, smoothing_steps: int) -> str:
    return f"tsvd{int(n_components)}_smooth{int(smoothing_steps)}"


def apply_tsvd_grid_spec(
    spec: NodePFNRunSpec, n_components: int, smoothing_steps: int
) -> NodePFNRunSpec:
    """Force TSVD at ``n_components`` (paper grid). Pads/clips inside preprocess."""
    tag = grid_cell_tag(n_components, smoothing_steps)
    return replace(
        spec,
        n_components=int(n_components),
        smoothing_steps=int(smoothing_steps),
        dim_reduction="tsvd",
        hparams_source=f"grid:{tag}",
    )


def featured_slugs_for_tsvd_grid() -> list[str]:
    """Featured multiclass paper NC graphs with no official ``run.sh`` mapping.

    Skips binaries, nofeat graphs, and NodePFN's 20-class cap (full-cora,
    hm-categories). Roman-empire is included even though it has an argparse
    extra spec — it is not in ``run.sh``.
    """
    from baselines.paths import resolve_nodepfn_data_root
    from baselines.registry import nc_eval_specs

    skip_class_cap = {"full-cora", "hm-categories"}
    data_root = resolve_nodepfn_data_root()
    slugs: list[str] = []
    for spec in nc_eval_specs(nofeat=False):
        if spec.is_binary or spec.ignore_features or is_nofeat_slug(spec.slug):
            continue
        if spec.slug in skip_class_cap:
            continue
        if spec.slug in NC_SLUG_TO_OFFICIAL:
            continue
        bundle = data_root / spec.slug
        if not (bundle / "features.npy").is_file():
            continue
        slugs.append(spec.slug)
    return slugs
