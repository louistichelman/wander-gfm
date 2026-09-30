"""Shared utilities for KG eval dataset loading."""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

Triple = Tuple[int, int, int]  # (head, relation, tail)

ARISTOV4_ZIP_URLS = [
    "https://zenodo.org/records/5942560/files/aristo-v4.zip",
    "https://zenodo.org/record/5942560/files/aristo-v4.zip?download=1",
]
MTDEA_ZIP_URL = "https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip"
NELL995_BASE_URL = (
    "https://raw.githubusercontent.com/LARS-research/RED-GNN/"
    "e064161e8d2b8b3e0ed17d7a746cbcb43b3c327e/transductive/data/nell"
)

# Registry key -> (zip family folder, trans/ind folder prefix)
MTDEA_REGISTRY: dict[str, tuple[str, str]] = {
    "WIKITOPICS_MT1_TAX": ("WikiTopics-MT1", "wikidata_taxv1"),
    "WIKITOPICS_MT1_HEALTH": ("WikiTopics-MT1", "wikidata_healthv1"),
    "WIKITOPICS_MT2_ORG": ("WikiTopics-MT2", "wikidata_orgv1"),
    "WIKITOPICS_MT2_SCI": ("WikiTopics-MT2", "wikidata_sciv1"),
    "WIKITOPICS_MT3_ART": ("WikiTopics-MT3", "wikidata_artv2"),
    "WIKITOPICS_MT3_INFRA": ("WikiTopics-MT3", "wikidata_infrav2"),
    "WIKITOPICS_MT4_SCI": ("WikiTopics-MT4", "wikidata_sciv2"),
    "WIKITOPICS_MT4_HEALTH": ("WikiTopics-MT4", "wikidata_healthv2"),
    "METAFAM": ("Metafam", "Metafam"),
    "FB_NELL": ("FBNELL", "FBNELL_v1"),
}
DACKGR_ZIP_URL = "https://raw.githubusercontent.com/THU-KEG/DacKGR/master/data.zip"


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None and len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def _is_non_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return 400 <= exc.response.status_code < 500
    return False


def ensure_download(
    url: str,
    dest: Path,
    *,
    retries: int = 1,
    timeout: tuple[int, int] = (30, 600),
    validate_zip: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not validate_zip or _is_valid_zip(dest):
            return
        print(f"Removing invalid cached file {dest}")
        dest.unlink()

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            print(f"Downloading {url} -> {dest}")
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fout:
                    for chunk in resp.iter_content(chunk_size=10 * 1024 * 1024):
                        if chunk:
                            fout.write(chunk)
            if validate_zip and not _is_valid_zip(dest):
                raise zipfile.BadZipFile(f"Downloaded file is not a valid zip: {dest}")
            return
        except Exception as exc:
            last_err = exc
            if dest.exists():
                dest.unlink()
            if _is_non_retryable_http_error(exc):
                break
            if attempt + 1 < retries:
                wait_s = min(15 * (2 ** attempt), 60)
                print(
                    f"Download failed ({exc}); retrying in {wait_s}s "
                    f"({attempt + 2}/{retries})..."
                )
                time.sleep(wait_s)
    if last_err is not None:
        raise last_err


def _download_first_available(urls: Iterable[str], dest: Path, **kwargs) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            ensure_download(url, dest, **kwargs)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(
        "All download URLs failed:\n" + "\n".join(f"  - {msg}" for msg in errors)
    )


def ensure_aristov4_raw(data_dir: Path) -> Path:
    """Download AristoV4 from Zenodo (Wander source) and materialize raw splits."""
    raw = data_dir / "AristoV4" / "raw"
    if (raw / "train.txt").exists():
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "aristo-v4.zip"
    try:
        _download_first_available(ARISTOV4_ZIP_URLS, zip_path, validate_zip=True, retries=3)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not download AristoV4 from Zenodo (often blocked or slow on HPC "
            "clusters). Place the extracted files manually at:\n"
            f"  {raw / 'train.txt'}\n"
            f"  {raw / 'valid.txt'}\n"
            f"  {raw / 'test.txt'}\n"
            "Source archive: https://zenodo.org/records/5942560/files/aristo-v4.zip"
        ) from exc
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw)
    for oldname, newname in [("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")]:
        src = raw / oldname
        if src.exists():
            src.rename(raw / newname)
    zip_path.unlink(missing_ok=True)
    return raw


def ensure_dackgr_raw(data_dir: Path, ds_folder: str) -> Path:
    """Extract DacKGR sparse-KG splits from the shared data.zip archive."""
    raw = data_dir / ds_folder / "raw"
    if (raw / "train.txt").exists():
        return raw

    base = data_dir / "SparseKG"
    zip_path = base / "data.zip"
    ensure_download(DACKGR_ZIP_URL, zip_path)

    extracted_root = base / "data"
    if not extracted_root.exists():
        base.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base)

    src_dir = extracted_root / ds_folder
    raw.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("train.triples", "train.txt"),
        ("dev.triples", "valid.txt"),
        ("test.triples", "test.txt"),
    ]:
        shutil.copy2(src_dir / src_name, raw / dst_name)
    return raw


def ensure_mtdea_raw(data_dir: Path, key: str) -> Path:
    """Materialize MTDEA / WikiTopics splits from the shared zip (Wander source)."""
    if key not in MTDEA_REGISTRY:
        raise KeyError(f"Unknown MTDEA dataset key: {key}")
    family, folder = MTDEA_REGISTRY[key]
    raw = data_dir / key / "default" / "raw"
    if (raw / "transductive_train.txt").exists():
        return raw

    base = data_dir / "MTDEA"
    zip_path = base / "MTDEA_data.zip"
    ensure_download(MTDEA_ZIP_URL, zip_path, validate_zip=True, retries=3)

    extracted = base / "MTDEA_datasets"
    if not extracted.exists():
        base.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base)

    src_trans = extracted / family / f"{folder}-trans"
    src_ind = extracted / family / f"{folder}-ind"
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_trans / "train.txt", raw / "transductive_train.txt")
    shutil.copy2(src_trans / "valid.txt", raw / "transductive_valid.txt")
    shutil.copy2(src_ind / "observe.txt", raw / "inference_graph.txt")
    shutil.copy2(src_ind / "test.txt", raw / "inf_test.txt")
    return raw


def ensure_nell995_raw(data_dir: Path) -> Path:
    """Download NELL995 from a pinned RED-GNN commit (main branch removed nell/)."""
    raw = data_dir / "NELL995" / "raw"
    if (raw / "train.txt").exists() and (raw / "facts.txt").exists():
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    for fname in ("facts.txt", "train.txt", "valid.txt", "test.txt"):
        ensure_download(f"{NELL995_BASE_URL}/{fname}", raw / fname)
    return raw


def load_triple_file(
    path: Path,
    *,
    delimiter: Optional[str] = None,
    htr_order: bool = False,
    inv_entity_vocab: Optional[Dict[str, int]] = None,
    inv_rel_vocab: Optional[Dict[str, int]] = None,
) -> Tuple[List[Triple], int, int, Dict[str, int], Dict[str, int]]:
    """Parse a triple file into integer (h, r, t) triples."""
    inv_entity_vocab = dict(inv_entity_vocab or {})
    inv_rel_vocab = dict(inv_rel_vocab or {})
    entity_cnt = len(inv_entity_vocab)
    rel_cnt = len(inv_rel_vocab)
    triples: List[Triple] = []

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split() if delimiter is None else line.split(delimiter)
            if len(parts) != 3:
                continue
            if htr_order:
                h_tok, t_tok, r_tok = parts
            else:
                h_tok, r_tok, t_tok = parts
            if h_tok not in inv_entity_vocab:
                inv_entity_vocab[h_tok] = entity_cnt
                entity_cnt += 1
            if t_tok not in inv_entity_vocab:
                inv_entity_vocab[t_tok] = entity_cnt
                entity_cnt += 1
            if r_tok not in inv_rel_vocab:
                inv_rel_vocab[r_tok] = rel_cnt
                rel_cnt += 1
            h = inv_entity_vocab[h_tok]
            t = inv_entity_vocab[t_tok]
            r = inv_rel_vocab[r_tok]
            triples.append((h, r, t))

    return triples, entity_cnt, rel_cnt, inv_entity_vocab, inv_rel_vocab


def entity_vocab_from_triple_files(
    paths: Iterable[Path],
    *,
    delimiter: Optional[str] = None,
    htr_order: bool = False,
) -> Tuple[int, Dict[str, int]]:
    """Assign entity ids in the same train→valid→test (then extra) order as the loaders."""
    inv_entity_vocab: Dict[str, int] = {}
    inv_rel_vocab: Dict[str, int] = {}
    entity_cnt = 0
    for path in paths:
        _, entity_cnt, _, inv_entity_vocab, inv_rel_vocab = load_triple_file(
            path,
            delimiter=delimiter,
            htr_order=htr_order,
            inv_entity_vocab=inv_entity_vocab,
            inv_rel_vocab=inv_rel_vocab,
        )
    del inv_rel_vocab
    return entity_cnt, inv_entity_vocab


def merge_triple_lists(*lists: Iterable[Triple]) -> List[Triple]:
    out: List[Triple] = []
    seen = set()
    for lst in lists:
        for tr in lst:
            if tr not in seen:
                seen.add(tr)
                out.append(tr)
    return out
