"""MinHash + HyperLogLog subgraph sketches (BUDDY / ELPH).

Port of ``subgraph-sketching/src/hashing.py`` without ``datasketch``,
``pandas``, or ``torch_sparse``. Node IDs are mixed with splitmix64; cardinality
uses standard HyperLogLog (linear counting for small sets) instead of the
HLL++ bias tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

HashTable = Dict[int, Dict[str, Tensor]]


@dataclass(frozen=True)
class HashConfig:
    max_hash_hops: int = 2
    minhash_num_perm: int = 128
    hll_p: int = 8
    use_zero_one: bool = False
    floor_sf: bool = False
    feature_batch_size: int = 1_000_000
    minhash_seed: int = 1


def _mix64(values: np.ndarray) -> np.ndarray:
    """Deterministic splitmix64 mix; ``values`` are treated as uint64."""
    x = values.astype(np.uint64, copy=True)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _hll_alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / float(m))


class MinhashPropagation(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="max")

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=-x)
        return -out


class HllPropagation(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="max")

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)


class ElphHashes:
    """Node-level sketches and pairwise subgraph features."""

    def __init__(self, cfg: HashConfig):
        if cfg.max_hash_hops not in {1, 2, 3}:
            raise ValueError(f"hashing is not implemented for {cfg.max_hash_hops} hops")
        self.max_hops = int(cfg.max_hash_hops)
        self.floor_sf = bool(cfg.floor_sf)
        self.use_zero_one = bool(cfg.use_zero_one)
        self.feature_batch_size = int(cfg.feature_batch_size)
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_minhash = np.uint64((1 << 32) - 1)
        self.minhash_seed = int(cfg.minhash_seed)
        self.num_perm = int(cfg.minhash_num_perm)
        self.minhash_prop = MinhashPropagation()
        self.p = int(cfg.hll_p)
        self.m = 1 << self.p
        self.alpha = _hll_alpha(self.m)
        self.max_rank = 64 - self.p
        self.hll_size = self.m
        self.hll_prop = HllPropagation()

    def num_struct_features(self) -> int:
        return self.max_hops * (self.max_hops + 2)

    def _init_permutations(self, num_perm: int) -> np.ndarray:
        gen = np.random.RandomState(self.minhash_seed)
        pairs = [
            (
                gen.randint(1, int(self._mersenne_prime), dtype=np.uint64),
                gen.randint(0, int(self._mersenne_prime), dtype=np.uint64),
            )
            for _ in range(num_perm)
        ]
        return np.array(pairs, dtype=np.uint64).T

    def _get_hll_rank(self, bits: np.ndarray) -> np.ndarray:
        flat = bits.astype(np.uint64, copy=False).ravel()
        bit_length = np.fromiter((int(v).bit_length() for v in flat), dtype=np.int32, count=flat.size)
        rank = self.max_rank - bit_length + 1
        if int(rank.min()) <= 0:
            raise ValueError(f"Hash value overflow, maximum size is {self.max_rank} bits")
        return rank.reshape(bits.shape)

    def initialise_minhash(self, n_nodes: int) -> Tensor:
        init_hv = np.ones((n_nodes, self.num_perm), dtype=np.uint64) * self._max_minhash
        a, b = self._init_permutations(self.num_perm)
        hv = _mix64(np.arange(1, n_nodes + 1, dtype=np.uint64))
        phv = (a * hv[:, None] + b) % self._mersenne_prime
        phv = np.bitwise_and(phv, self._max_minhash)
        return torch.from_numpy(np.minimum(phv, init_hv).astype(np.int64))

    def initialise_hll(self, n_nodes: int) -> Tensor:
        regs = np.zeros((n_nodes, self.m), dtype=np.int8)
        hv = _mix64(np.arange(1, n_nodes + 1, dtype=np.uint64))
        reg_index = (hv & np.uint64(self.m - 1)).astype(np.int64)
        bits = hv >> np.uint64(self.p)
        ranks = self._get_hll_rank(bits)
        rows = np.arange(n_nodes)
        regs[rows, reg_index] = np.maximum(regs[rows, reg_index], ranks.astype(np.int8))
        return torch.from_numpy(regs)

    def build_hash_tables(self, num_nodes: int, edge_index: Tensor) -> Tuple[HashTable, Tensor]:
        hash_edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        cards = torch.zeros((num_nodes, self.max_hops), dtype=torch.float32, device=edge_index.device)
        tables: HashTable = {}
        for k in range(self.max_hops + 1):
            start = time()
            if k == 0:
                tables[k] = {
                    "minhash": self.initialise_minhash(num_nodes).to(edge_index.device),
                    "hll": self.initialise_hll(num_nodes).to(edge_index.device),
                }
            else:
                tables[k] = {
                    "hll": self.hll_prop(tables[k - 1]["hll"], hash_edge_index),
                    "minhash": self.minhash_prop(tables[k - 1]["minhash"], hash_edge_index),
                }
                cards[:, k - 1] = self.hll_count(tables[k]["hll"])
            _ = start
        return tables, cards

    def hll_count(self, regs: Tensor) -> Tensor:
        if regs.dim() == 1:
            regs = regs.unsqueeze(0)
        regs_f = regs.to(torch.float32)
        estimate = (self.alpha * float(self.m) ** 2) / torch.sum(2.0 ** (-regs_f), dim=1)
        num_zero = self.m - torch.count_nonzero(regs, dim=1)
        small = estimate <= (2.5 * float(self.m))
        use_lc = small & (num_zero > 0)
        if bool(use_lc.any()):
            lc = float(self.m) * torch.log(float(self.m) / num_zero.clamp(min=1).to(torch.float32))
            estimate = torch.where(use_lc, lc, estimate)
        return estimate

    def jaccard(self, src: Tensor, dst: Tensor) -> Tensor:
        if src.shape != dst.shape:
            raise ValueError("source and destination hash value shapes must be the same")
        return torch.count_nonzero(src == dst, dim=-1).to(torch.float32) / float(self.num_perm)

    def _get_intersections(self, edge_list: Tensor, hash_table: HashTable) -> Dict[Tuple[int, int], Tensor]:
        intersections: Dict[Tuple[int, int], Tensor] = {}
        for k1 in range(1, self.max_hops + 1):
            for k2 in range(1, self.max_hops + 1):
                src_hll = hash_table[k1]["hll"][edge_list[:, 0]]
                src_minhash = hash_table[k1]["minhash"][edge_list[:, 0]]
                dst_hll = hash_table[k2]["hll"][edge_list[:, 1]]
                dst_minhash = hash_table[k2]["minhash"][edge_list[:, 1]]
                jaccard = self.jaccard(src_minhash, dst_minhash)
                unions = torch.maximum(src_hll, dst_hll)
                intersections[(k1, k2)] = jaccard * self.hll_count(unions)
        return intersections

    def _features_from_intersections(
        self,
        intersections: Dict[Tuple[int, int], Tensor],
        cards1: Tensor,
        cards2: Tensor,
        n: int,
        device: torch.device,
    ) -> Tensor:
        features = torch.zeros((n, self.num_struct_features()), dtype=torch.float32, device=device)
        features[:, 0] = intersections[(1, 1)]
        if self.max_hops == 1:
            features[:, 1] = cards2[:, 0] - features[:, 0]
            features[:, 2] = cards1[:, 0] - features[:, 0]
        elif self.max_hops == 2:
            features[:, 1] = intersections[(2, 1)] - features[:, 0]
            features[:, 2] = intersections[(1, 2)] - features[:, 0]
            features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]
            features[:, 4] = cards2[:, 0] - torch.sum(features[:, 0:2], dim=1)
            features[:, 5] = cards1[:, 0] - features[:, 0] - features[:, 2]
            features[:, 6] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1)
            features[:, 7] = cards1[:, 1] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]
        elif self.max_hops == 3:
            features[:, 1] = intersections[(2, 1)] - features[:, 0]
            features[:, 2] = intersections[(1, 2)] - features[:, 0]
            features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]
            features[:, 4] = intersections[(3, 1)] - features[:, 0] - features[:, 1]
            features[:, 5] = intersections[(1, 3)] - features[:, 0] - features[:, 2]
            features[:, 6] = intersections[(3, 2)] - torch.sum(features[:, 0:4], dim=1) - features[:, 4]
            features[:, 7] = intersections[(2, 3)] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]
            features[:, 8] = intersections[(3, 3)] - torch.sum(features[:, 0:8], dim=1)
            features[:, 9] = cards2[:, 0] - features[:, 0] - features[:, 1] - features[:, 4]
            features[:, 10] = cards1[:, 0] - features[:, 0] - features[:, 2] - features[:, 5]
            features[:, 11] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 6] - features[:, 9]
            features[:, 12] = cards1[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 7] - features[:, 10]
            features[:, 13] = cards2[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 9] - features[:, 11]
            features[:, 14] = cards1[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 10] - features[:, 12]
        if not self.use_zero_one:
            if self.max_hops == 2:
                features[:, 4] = 0
                features[:, 5] = 0
            elif self.max_hops == 3:
                features[:, 4] = 0
                features[:, 5] = 0
                features[:, 11] = 0
                features[:, 12] = 0
        if self.floor_sf:
            features.clamp_(min=0)
        return features

    def get_subgraph_features(
        self,
        links: Tensor,
        hash_table: HashTable,
        cards: Tensor,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        if links.dim() == 1:
            links = links.unsqueeze(0)
        if links.numel() == 0:
            return torch.zeros((0, self.num_struct_features()), dtype=torch.float32, device=links.device)
        bs = int(batch_size or self.feature_batch_size)
        parts = []
        cards = cards.to(links.device)
        for start in range(0, links.size(0), bs):
            batch = links[start : start + bs]
            intersections = self._get_intersections(batch, hash_table)
            cards1 = cards[batch[:, 0]]
            cards2 = cards[batch[:, 1]]
            parts.append(
                self._features_from_intersections(
                    intersections, cards1, cards2, batch.size(0), batch.device
                )
            )
        return torch.cat(parts, dim=0)


def move_hash_tables(tables: HashTable, device: torch.device) -> HashTable:
    return {k: {name: tensor.to(device) for name, tensor in hop.items()} for k, hop in tables.items()}
