"""Register all KG eval datasets into DATASET_REGISTRY."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from ...graph_bundle import GraphBundle
from .bundle_builder import load_three_split_transductive
from .common import ensure_aristov4_raw, ensure_dackgr_raw, ensure_mtdea_raw, ensure_nell995_raw
from .conceptnet_feat import load_conceptnet_feat_bundle
from .inductive import load_generic_inductive_bundle, load_grail_bundle, load_hm_bundle
from .wn_feat import load_wn18rr_feat_bundle, load_wn18rr_ind_feat_bundle

_KG = dict(
    HAS_NODE_FEATURES=False,
    HAS_NODE_LABELS=False,
    HAS_PREDEFINED_NODE_SPLIT=False,
    HAS_PREDEFINED_EDGE_SPLIT=True,
    IS_KNOWLEDGE_GRAPH=True,
    IS_INDUCTIVE=False,
    INDUCTIVE_FILTER_MODE="transductive",
    EDGE_SET_MODE="train_only",
    # KG benchmarks are compared with MRR including head predictions.
    PREFERRED_LINK_PRED_EVAL="mrr",
    PREFERRED_EVALUATE_HEAD_PREDICTIONS=True,
)


def _mod(
    name: str,
    loader: Callable[..., GraphBundle],
    *,
    is_inductive: bool = False,
    inductive_filter_mode: str = "transductive",
    supported_versions: Optional[List[str]] = None,
    edge_set_mode: str = "train_only",
) -> ModuleType:
    m = ModuleType(name)
    m.NAME = name
    m.load_graph_bundle = loader
    m.load_base_dataset = None  # type: ignore[assignment]
    for k, v in _KG.items():
        setattr(m, k, v)
    m.IS_INDUCTIVE = is_inductive
    m.INDUCTIVE_FILTER_MODE = inductive_filter_mode
    m.SUPPORTED_VERSIONS = supported_versions
    m.EDGE_SET_MODE = edge_set_mode
    return m


def _trans(
    key: str,
    name: str,
    urls: List[str],
    files: List[str],
    **kwargs,
) -> ModuleType:
    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ) -> GraphBundle:
        del seed, dataset_version
        return load_three_split_transductive(
            Path(data_dir), name, urls, files, add_inverse_edges_kgs, **kwargs
        )

    return _mod(key, loader)


def _feat_mod(
    key: str,
    display_name: str,
    loader: Callable[..., GraphBundle],
    edge_set_mode: str = "train_only",
) -> ModuleType:
    m = _mod(key, loader, edge_set_mode=edge_set_mode)
    m.HAS_NODE_FEATURES = True
    m.NAME = display_name
    return m


def _grail(key: str, name: str, prefix: str) -> ModuleType:
    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ) -> GraphBundle:
        del seed
        version = dataset_version or "v1"
        return load_grail_bundle(
            Path(data_dir), name, prefix, version, add_inverse_edges_kgs
        )

    return _mod(
        key, loader,
        is_inductive=True,
        inductive_filter_mode="grail",
        supported_versions=["v1", "v2", "v3", "v4"],
    )


def _inductive4(
    key: str,
    name: str,
    urls: List[str],
    files: List[str],
    valid_on_inf: bool,
    filter_mode: str,
    versions: List[str],
) -> ModuleType:
    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ) -> GraphBundle:
        del seed
        version = dataset_version or versions[0]
        return load_generic_inductive_bundle(
            Path(data_dir), name, urls, files, version,
            add_inverse_edges_kgs, valid_on_inf, filter_mode,
        )

    return _mod(
        key, loader,
        is_inductive=True,
        inductive_filter_mode=filter_mode,
        supported_versions=versions,
    )


def _hm(key: str, name: str, folder: str) -> ModuleType:
    def loader(
        data_dir: str,
        seed: int = 42,
        add_inverse_edges_kgs: bool = True,
        dataset_version: Optional[str] = None,
    ) -> GraphBundle:
        del seed
        version = dataset_version or "1k"
        return load_hm_bundle(
            Path(data_dir), name, version, folder, add_inverse_edges_kgs
        )

    return _mod(
        key, loader,
        is_inductive=True,
        inductive_filter_mode="hm_mtdea",
        supported_versions=["1k", "3k", "5k", "indigo"],
    )


def build_kg_eval_registry() -> Dict[str, ModuleType]:
    reg: Dict[str, ModuleType] = {}

    codex_base = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-{}/{}"
    for key, slug, disp in [
        ("CODEX_SMALL", "s", "CoDEx-Small"),
        ("CODEX_LARGE", "l", "CoDEx-Large"),
    ]:
        reg[key] = _trans(
            key, disp,
            [codex_base.format(slug, s) for s in ("train.txt", "valid.txt", "test.txt")],
            ["train.txt", "valid.txt", "test.txt"],
        )

    reg["YAGO310"] = _trans(
        "YAGO310", "YAGO3-10",
        [
            "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
            "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
            "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
        ],
        ["train.txt", "valid.txt", "test.txt"],
    )

    reg["DBPEDIA100K"] = _trans(
        "DBPEDIA100K", "DBpedia100k",
        [
            "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
            "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
            "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
        ],
        ["train.txt", "valid.txt", "test.txt"],
    )

    reg["CONCEPTNET100K"] = _trans(
        "CONCEPTNET100K", "ConceptNet100k",
        [
            "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
            "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
            "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
        ],
        ["train.txt", "valid.txt", "test.txt"],
        delimiter="\t",
    )

    def _conceptnet_feat_loader(
        data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
    ):
        del seed, dataset_version
        return load_conceptnet_feat_bundle(data_dir, add_inverse_edges_kgs)

    reg["CONCEPTNET100K_FEAT"] = _feat_mod(
        "CONCEPTNET100K_FEAT", "ConceptNet100k-Feat", _conceptnet_feat_loader
    )

    reg["HETIONET"] = _trans(
        "HETIONET", "Hetionet",
        [
            "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
            "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
            "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        ],
        ["train.txt", "valid.txt", "test.txt"],
    )

    def _aristov4_loader(
        data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
    ):
        del seed, dataset_version
        raw = ensure_aristov4_raw(Path(data_dir))
        return load_three_split_transductive(
            Path(data_dir),
            "AristoV4",
            [],
            ["train.txt", "valid.txt", "test.txt"],
            add_inverse_edges_kgs,
            delimiter="\t",
            raw_paths=[raw / "train.txt", raw / "valid.txt", raw / "test.txt"],
        )

    reg["ARISTOV4"] = _mod("ARISTOV4", _aristov4_loader)

    def _nell_loader(
        data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
    ):
        del seed, dataset_version
        raw = ensure_nell995_raw(Path(data_dir))
        return load_three_split_transductive(
            Path(data_dir),
            "NELL995",
            [],
            ["train.txt", "valid.txt", "test.txt"],
            add_inverse_edges_kgs,
            extra_train_context_files=["facts.txt"],
            raw_paths=[raw / "train.txt", raw / "valid.txt", raw / "test.txt"],
        )

    reg["NELL995"] = _mod("NELL995", _nell_loader)

    for key, sub in [
        ("WD_SINGER", "WD-singer"),
        ("NELL23K", "NELL23K"),
        ("FB15K237_10", "FB15K-237-10"),
        ("FB15K237_20", "FB15K-237-20"),
        ("FB15K237_50", "FB15K-237-50"),
    ]:
        def _dackgr_loader(
            data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
            _sub=sub, _key=key,
        ):
            del seed, dataset_version
            raw = ensure_dackgr_raw(Path(data_dir), _sub)
            return load_three_split_transductive(
                Path(data_dir),
                _sub,
                [],
                ["train.txt", "valid.txt", "test.txt"],
                add_inverse_edges_kgs,
                delimiter="\t",
                htr_order=True,
                raw_paths=[raw / "train.txt", raw / "valid.txt", raw / "test.txt"],
            )

        reg[key] = _mod(key, _dackgr_loader)

    def _wn18rr_feat_loader(
        data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
    ):
        del seed, dataset_version
        return load_wn18rr_feat_bundle(data_dir, add_inverse_edges_kgs)

    reg["WN18RR_FEAT"] = _feat_mod(
        "WN18RR_FEAT",
        "WN18RR-Feat",
        _wn18rr_feat_loader,
        edge_set_mode="all_positives",
    )

    # GraIL inductive
    for key, disp, prefix in [
        ("FB15K237_IND", "FB15k237-Ind", "fb237"),
        ("WN18RR_IND", "WN18RR-Ind", "WN18RR"),
        ("NELL_IND", "NELL-Ind", "nell"),
    ]:
        reg[key] = _grail(key, disp, prefix)

    def _wn18rr_ind_feat_loader(
        data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
    ):
        del seed
        version = dataset_version or "v1"
        return load_wn18rr_ind_feat_bundle(
            data_dir, version, add_inverse_edges_kgs
        )

    reg["WN18RR_IND_FEAT"] = _mod(
        "WN18RR_IND_FEAT",
        _wn18rr_ind_feat_loader,
        is_inductive=True,
        inductive_filter_mode="grail",
        supported_versions=["v1", "v2", "v3", "v4"],
    )
    reg["WN18RR_IND_FEAT"].HAS_NODE_FEATURES = True
    reg["WN18RR_IND_FEAT"].NAME = "WN18RR-Ind-Feat"

    ilpc_urls = [
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/train.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_validation.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_test.txt",
    ]
    ilpc_files = ["train.txt", "inference.txt", "inf_valid.txt", "inf_test.txt"]
    for key, ver in [("ILPC2022_SMALL", "small"), ("ILPC2022_LARGE", "large")]:
        reg[key] = _inductive4(
            key, f"ILPC2022-{ver}", ilpc_urls, ilpc_files,
            valid_on_inf=True, filter_mode="ilpc", versions=[ver],
        )

    ingram_urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/{fam}-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/{fam}-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/{fam}-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/{fam}-%s/test.txt",
    ]
    ingram_files = ["train.txt", "msg.txt", "valid.txt", "test.txt"]
    for key, fam, vers in [
        ("FB_INGRAM", "FB", ["25", "50", "75", "100"]),
        ("WK_INGRAM", "WK", ["25", "50", "75", "100"]),
        ("NL_INGRAM", "NL", ["0", "25", "50", "75", "100"]),
    ]:
        urls = [u.format(fam=fam) for u in ingram_urls]
        reg[key] = _inductive4(
            key, f"{fam}-InGram", urls, ingram_files,
            valid_on_inf=True, filter_mode="ilpc", versions=vers,
        )

    for key, folder in [
        ("HM_1K", "Hamaguchi-BM_both-1000"),
        ("HM_3K", "Hamaguchi-BM_both-3000"),
        ("HM_5K", "Hamaguchi-BM_both-5000"),
        ("HM_INDIGO", "INDIGO-BM"),
    ]:
        reg[key] = _hm(key, f"HM-{folder}", folder)

    # MTDEA / WikiTopics — shared zip (per-file S3 URLs return 403)
    for key in [
        "WIKITOPICS_MT1_TAX",
        "WIKITOPICS_MT1_HEALTH",
        "WIKITOPICS_MT2_ORG",
        "WIKITOPICS_MT2_SCI",
        "WIKITOPICS_MT3_ART",
        "WIKITOPICS_MT3_INFRA",
        "WIKITOPICS_MT4_SCI",
        "WIKITOPICS_MT4_HEALTH",
        "METAFAM",
        "FB_NELL",
    ]:
        def _mtdea_loader(
            data_dir, seed=42, add_inverse_edges_kgs=True, dataset_version=None,
            _key=key,
        ):
            del seed, dataset_version
            ensure_mtdea_raw(Path(data_dir), _key)
            return load_generic_inductive_bundle(
                Path(data_dir), _key,
                ["local"] * 4,
                ["transductive_train.txt", "inference_graph.txt",
                 "transductive_valid.txt", "inf_test.txt"],
                "default", add_inverse_edges_kgs, False, "hm_mtdea",
            )

        reg[key] = _mod(
            key, _mtdea_loader,
            is_inductive=True,
            inductive_filter_mode="hm_mtdea",
            supported_versions=None,
        )

    return reg


def register_kg_eval_datasets(registry: Dict[str, Any]) -> None:
    """Merge KG eval modules into the global DATASET_REGISTRY."""
    registry.update(build_kg_eval_registry())
