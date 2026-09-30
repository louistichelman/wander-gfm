"""Dataset registry for Wander."""

from . import (
    cora, citeseer, pubmed, FB15k_237, WN18RR, CoDEx_Medium,
    roman_empire, amazon_ratings,
    photo, texas, actor, usa,
    chameleon_filtered, squirrel_filtered, cornell, wisconsin, full_cora, full_dblp, wiki_cs,
    co_cs, co_physics, brazil, europe, computers,
    hm_categories, pokec_regions, pokec_regions_100k_top10,
    cycles,
    grids,
    city_paris, city_shanghai, city_la,
)

DATASET_REGISTRY = {
    "CORA": cora,
    "CITESEER": citeseer,
    "PUBMED": pubmed,
    "FB15K_237": FB15k_237,
    "WN18RR": WN18RR,
    "CODEX_MEDIUM": CoDEx_Medium,
    "ROMAN_EMPIRE": roman_empire,
    "AMAZON_RATINGS": amazon_ratings,
    "PHOTO": photo,
    "TEXAS": texas,
    "ACTOR": actor,
    "USA": usa,
    "CHAMELEON_FILTERED": chameleon_filtered,
    "SQUIRREL_FILTERED": squirrel_filtered,
    "CORNELL": cornell,
    "WISCONSIN": wisconsin,
    "FULL_CORA": full_cora,
    "FULL_DBLP": full_dblp,
    "WIKI_CS": wiki_cs,
    "CO_CS": co_cs,
    "CO_PHYSICS": co_physics,
    "BRAZIL": brazil,
    "EUROPE": europe,
    "COMPUTERS": computers,
    "HM_CATEGORIES": hm_categories,
    "POKEC_REGIONS": pokec_regions,
    "POKEC_REGIONS_100K_TOP10": pokec_regions_100k_top10,
    "CYCLES": cycles,
    "GRIDS": grids,
    "CITY_PARIS": city_paris,
    "CITY_SHANGHAI": city_shanghai,
    "CITY_LA": city_la,
}

from .kg_eval import register_kg_eval_datasets

register_kg_eval_datasets(DATASET_REGISTRY)

from .anygraph_lp import register_anygraph_datasets

register_anygraph_datasets(DATASET_REGISTRY)

from .unilp_lp import register_unilp_datasets

register_unilp_datasets(DATASET_REGISTRY)
