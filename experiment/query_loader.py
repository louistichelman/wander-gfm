"""Query loader for Wander."""

import os
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING
from torch import Tensor
from torch_geometric.data import Data
import torch

if TYPE_CHECKING:
    from data.graph_bundle import GraphBundle, SplitName

from data.eval_subsample import (
    dataset_eval_subsample_seed as _dataset_eval_subsample_seed,
    subsample_indices as _subsample_indices,
)
from data.sampled_recall import (
    DEFAULT_NUM_NEG,
    load_or_build_sampled_recall_cache,
)
from data.graph_bundle import (
    GraphBundle,
    SplitName,
    resolve_split,
    query_mask_for_split,
    is_link_prediction,
)

StepItem = Tuple[str, Tensor, Data]
LoaderYield = Union[StepItem, List[StepItem]]


def _cpu_batch_rng(batch_seed: Optional[int]) -> Optional[torch.Generator]:
    if batch_seed is None:
        return None
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(batch_seed))
    return rng


def _randperm_index(n: int, rng: Optional[torch.Generator]) -> Tensor:
    if rng is None:
        return torch.arange(n)
    return torch.randperm(n, generator=rng)


def _build_bfs_proximity_batches_new_seeds(
    edge_index: Tensor,
    num_nodes: int,
    target_indices: Tensor,
    batch_size: int,
    max_radius: Optional[int] = None,
    batch_seed: Optional[int] = None,
) -> List[Tensor]:
    """Partition ``target_indices`` into packed BFS balls around new seeds.

    A full batch never continues the previous BFS frontier. Each seed grows a
    compact ball (optional ``max_radius``) over the original undirected graph
    (paths may go through non-targets / already-taken nodes) but only
    uncovered targets are collected. If the ball fills the batch, leftover
    hits in that layer stay uncovered and the next batch starts at a new
    max-degree seed. If the ball dies short of ``batch_size``, another seed
    is packed into the same batch. All batches except possibly the last have
    size ``batch_size``.

    If ``batch_seed`` is set, max-degree ties and last-layer overflow leftover
    (which stays uncovered) are shuffled so different seeds produce different
    partitions of the same set.

    Returns a list of index tensors; their concatenation is exactly
    ``target_indices`` (as a set), with no duplicates. Deterministic given the
    graph, target set, ``max_radius``, and ``batch_seed``.
    """
    edge_index = edge_index.detach().cpu()
    target_indices = target_indices.detach().cpu()
    rng = _cpu_batch_rng(batch_seed)

    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])
    order = torch.argsort(src)
    src = src[order]
    dst = dst[order]
    degrees = torch.bincount(src, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indptr[1:] = torch.cumsum(degrees, dim=0)

    uncovered = torch.zeros(num_nodes, dtype=torch.bool)
    uncovered[target_indices] = True
    remaining = int(uncovered.sum().item())

    max_visited_per_seed = max(batch_size * 64, 8192)

    batches: List[Tensor] = []
    current: List[Tensor] = []
    current_count = 0
    visited = torch.zeros(num_nodes, dtype=torch.bool)

    def _flush() -> None:
        nonlocal current, current_count
        if current_count > 0:
            batches.append(torch.cat(current))
            current = []
            current_count = 0

    while remaining > 0:
        cand = uncovered.nonzero(as_tuple=True)[0]
        cand = cand[_randperm_index(int(cand.numel()), rng)]
        seed = cand[degrees[cand].argmax()]

        visited.fill_(False)
        visited[seed] = True
        frontier = seed.view(1)
        visited_count = 1
        hop = 0
        progressed = False

        while frontier.numel() > 0:
            hits = frontier[uncovered[frontier]]
            if hits.numel() > 0:
                hits = hits[_randperm_index(int(hits.numel()), rng)]
                take = min(int(hits.numel()), batch_size - current_count)
                chosen = hits[:take]
                uncovered[chosen] = False
                remaining -= take
                current.append(chosen)
                current_count += take
                progressed = True
                if current_count >= batch_size:
                    _flush()
                    # Do not continue this frontier into the next batch.
                    break
            if remaining == 0:
                break
            if max_radius is not None and hop >= max_radius:
                break
            if visited_count >= max_visited_per_seed:
                break
            counts = degrees[frontier]
            total = int(counts.sum().item())
            if total == 0:
                break
            starts = indptr[frontier]
            offs = torch.arange(total) - torch.repeat_interleave(
                torch.cumsum(counts, dim=0) - counts, counts
            )
            neigh = dst[torch.repeat_interleave(starts, counts) + offs]
            neigh = torch.unique(neigh[~visited[neigh]])
            visited[neigh] = True
            visited_count += int(neigh.numel())
            frontier = neigh
            hop += 1

        if not progressed:
            # Unreachable if the seed is uncovered; keep the loop from stalling.
            uncovered[seed] = False
            remaining -= 1
            current.append(seed.view(1))
            current_count += 1
            if current_count >= batch_size:
                _flush()

    _flush()
    return batches


def _chunk_indices(indices: Tensor, batch_size: int) -> List[Tensor]:
    """Split node/query indices into fixed-size batches."""
    if indices.numel() == 0:
        return []
    batches: List[Tensor] = []
    for start in range(0, int(indices.numel()), batch_size):
        batches.append(indices[start : start + batch_size])
    return batches


def _ddp_padded_batch_shard(
    batches: List[Tensor],
    rank: int,
    world_size: int,
) -> List[Tensor]:
    """Round-robin shard batches across DDP ranks with equal per-rank counts.

    Eval collectives (``gather_object`` / ``all_reduce``) require every rank to
    finish the batch loop together. Round-robin sharding alone can leave ranks
    with different batch counts when the global batch count is not divisible by
    ``world_size``; pad with empty index tensors so all ranks iterate equally.
    """
    if world_size <= 1:
        return batches
    empty = batches[0][:0] if batches else torch.empty(0, dtype=torch.long)
    if not batches:
        return [empty]
    max_per_rank = (len(batches) + world_size - 1) // world_size
    mine = list(batches[rank::world_size])
    while len(mine) < max_per_rank:
        mine.append(empty)
    return mine


def _inverse_queries(forward_queries: Tensor, num_relations: int) -> Tensor:
    """Map (head, rel, tail) queries to (tail, rel + num_relations//2, head)."""
    inv = forward_queries.clone()
    inv[:, 0] = forward_queries[:, 2]
    inv[:, 2] = forward_queries[:, 0]
    inv[:, 1] = (forward_queries[:, 1] + num_relations // 2) % num_relations
    return inv


def _with_predict_head(queries: Tensor, predict_head: int) -> Tensor:
    """Append a predict_head column (0=tail query, 1=head query)."""
    flag = torch.full((queries.shape[0], 1), predict_head, dtype=queries.dtype, device=queries.device)
    return torch.cat([queries, flag], dim=1)


def _direction_head_queries(forward_queries: Tensor) -> Tensor:
    """Same (h, r, t) triples flagged as head-prediction queries."""
    return _with_predict_head(forward_queries, 1)

class QueryLoaderLinkPrediction:
    """Custom loader that batches link prediction queries with proper edge masking.
    
    In training mode, batch edges are masked from the visible graph to prevent
    information leakage. In eval mode, all visible edges are available.
    """
    
    def __init__(
        self,
        data: Data,
        mask: Tensor,
        batch_size: int,
        shuffle: bool = True,
        training: bool = True,
        rank: int = 0,
        world_size: int = 1,
        model_type: str = "wander",
        max_eval_samples: Optional[int] = None,
        train_edge_set=None,
        use_direction_head_queries: bool = False,
        evaluate_head_predictions: bool = False,
        eval_subsample_seed: Optional[int] = None,
        train_without_replacement: bool = False,
    ):
        """Initialize the QueryLoaderLinkPrediction.
        
        Args:
            data: PyG Data object with edge_index, edge_type, visible_mask, etc.
            mask: Boolean mask indicating which edges to sample queries from.
            batch_size: Queries per batch.
            shuffle: Whether to shuffle queries (typically True for training).
            training: If True, mask batch edges from visible graph.
            rank: GPU rank for distributed training (0 for single-GPU).
            world_size: Total number of GPUs (1 for single-GPU).
            model_type: Type of model to use ("wander" or "Flock").
            max_eval_samples: Max link queries per eval pass when iterating
                (``None`` = use all queries on this rank).
            evaluate_head_predictions: Eval only. If True, evaluate BOTH tail and
                head predictions for every test triple (forward queries plus
                inverse-relation queries; or predict_head=0 and predict_head=1
                queries when using direction head queries). If False, evaluate only
                tail predictions (forward queries).
            train_without_replacement: If True (training), consume a shuffled
                permutation of this rank's queries without replacement. The deck
                continues across epochs and is reshuffled only when empty.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.model_type = model_type
        self.max_eval_samples = max_eval_samples
        self._train_edge_set = train_edge_set
        self.use_direction_head_queries = use_direction_head_queries
        self.evaluate_head_predictions = evaluate_head_predictions
        self._rank = int(rank)
        self._world_size = int(world_size)

        all_indices = mask.nonzero(as_tuple=True)[0]
        self._training_set_size = 1#len(all_indices)
        # Eval on all queries globally, then shard batches (not queries) across
        # DDP ranks so every rank runs the same number of eval steps.
        if world_size > 1 and not training:
            self.query_edge_indices = all_indices
        elif world_size > 1:
            self.query_edge_indices = all_indices[rank::world_size]
        else:
            self.query_edge_indices = all_indices
        
        edge_index = data.edge_index
        edge_type = data.edge_type
        forward_queries = torch.stack([
            edge_index[0, self.query_edge_indices],
            edge_type[self.query_edge_indices],
            edge_index[1, self.query_edge_indices],
        ], dim=1)

        self._eval_bidirectional = (
            not training and getattr(data, "has_inverse_edges", False)
        )
        if self._eval_bidirectional:
            if evaluate_head_predictions:
                # Evaluate BOTH directions: tail prediction (forward queries) and
                # head prediction (inverse-relation queries).
                self.queries = torch.cat(
                    [
                        forward_queries,
                        _inverse_queries(forward_queries, data.num_relations),
                    ],
                    dim=0,
                )
            else:
                self.queries = forward_queries
        elif not training and use_direction_head_queries:
            if evaluate_head_predictions:
                # Evaluate BOTH directions via the predict_head flag on the same
                # triple: predict_head=0 (tail) and predict_head=1 (head).
                self.queries = torch.cat(
                    [
                        _with_predict_head(forward_queries, 0),
                        _direction_head_queries(forward_queries),
                    ],
                    dim=0,
                )
            else:
                self.queries = _with_predict_head(forward_queries, 0)
        elif use_direction_head_queries:
            self.queries = _with_predict_head(forward_queries, 0)
        else:
            self.queries = forward_queries

        self.num_queries = self.queries.shape[0]

        self._eval_subsample_seed = eval_subsample_seed
        self._eval_query_perm: Optional[Tensor] = None
        self._ddp_eval_batches: Optional[List[Tensor]] = None
        if max_eval_samples is not None and eval_subsample_seed is not None:
            if world_size > 1 and not training:
                cap = max_eval_samples * world_size
                k = min(cap, self.num_queries)
            else:
                k = min(max_eval_samples, self.num_queries)
            g = torch.Generator()
            g.manual_seed(eval_subsample_seed)
            perm = torch.randperm(self.num_queries, generator=g)[:k]
            if world_size > 1 and not training:
                batch_slices = _chunk_indices(perm, self.batch_size)
                self._ddp_eval_batches = _ddp_padded_batch_shard(
                    batch_slices, rank, world_size,
                )
            else:
                self._eval_query_perm = perm

        self.train_without_replacement = (
            bool(train_without_replacement) and bool(training)
        )
        self._train_perm: Optional[Tensor] = None
        self._train_cursor = 0
        self._reshuffle_count = 0

    def _reshuffle_train_perm(self) -> None:
        """Draw a new no-replacement deck. Does not run at epoch boundaries."""
        if self.num_queries <= 0:
            self._train_perm = torch.empty(0, dtype=torch.long)
            self._train_cursor = 0
            return
        g = torch.Generator()
        g.manual_seed(int(self._reshuffle_count) + 10007 * int(self._rank))
        self._train_perm = torch.randperm(self.num_queries, generator=g)
        self._train_cursor = 0
        self._reshuffle_count += 1

    def set_epoch(self, epoch: int) -> None:
        """Ensure a no-replacement deck exists; do not reset the cursor."""
        if not self.train_without_replacement:
            return
        if self._train_perm is None:
            self._reshuffle_train_perm()

    def remaining_train_queries(self) -> int:
        """Unconsumed queries in the current no-replacement deck."""
        if not self.train_without_replacement:
            return self.num_queries
        if self._train_perm is None:
            return self.num_queries
        return max(0, int(self.num_queries) - int(self._train_cursor))
    
    def __len__(self) -> int:
        """Number of batches."""
        if self._ddp_eval_batches is not None:
            return len(self._ddp_eval_batches)
        n = (
            int(self._eval_query_perm.numel())
            if self._eval_query_perm is not None
            else self.num_queries
        )
        if self.max_eval_samples is not None and self._eval_query_perm is None:
            n = min(n, self.max_eval_samples)
        return (n + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Data]]:
        """Iterate over batches, yielding (batch_queries, batch_data) tuples.
        
        Only used during evaluation. During training we use sample_batch instead.
        Subsamples to at most ``max_eval_samples`` queries on this rank. When
        ``_eval_query_perm`` was built at init, the same queries are used every epoch.
        """
        if self._ddp_eval_batches is not None:
            n_forward = len(self.query_edge_indices)
            for batch_indices in self._ddp_eval_batches:
                if batch_indices.numel() == 0:
                    yield torch.empty((0, self.queries.shape[1]), dtype=self.queries.dtype), self._get_visible_data()
                    continue
                batch_queries = self.queries[batch_indices]
                batch_edge_indices = self.query_edge_indices[batch_indices % n_forward]
                yield batch_queries, self._get_visible_data()
            return
        if self._eval_query_perm is not None:
            perm = self._eval_query_perm
        else:
            k = self.num_queries
            if self.max_eval_samples is not None:
                k = min(k, self.max_eval_samples)
            perm = torch.randperm(self.num_queries)[:k]

        for start_idx in range(0, len(perm), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(perm))
            batch_indices = perm[start_idx:end_idx]
            
            # Get batch queries
            batch_queries = self.queries[batch_indices]
            n_forward = len(self.query_edge_indices)
            batch_edge_indices = self.query_edge_indices[batch_indices % n_forward]
            
            if self.training:
                batch_data = self._mask_batch_edges(batch_edge_indices)
            else:
                batch_data = self._get_visible_data()
            
            yield batch_queries, batch_data
    
    def sample_batch(self) -> Tuple[Tensor, Data]:
        """Sample a single training batch.

        Default: with replacement via ``torch.randint``.

        When ``train_without_replacement`` is True: consume the next slice of
        a shuffled deck that continues across epochs. The last batch of a pass
        may be shorter; the deck is reshuffled only when it is empty.

        Otherwise: ``batch_size`` mixed-relation queries. When
        ``data.has_inverse_edges`` is True, exactly half of the sampled queries
        are reversed. When ``use_direction_head_queries`` is True, half the batch
        gets predict_head=1.

        Returns:
            Tuple of (batch_queries, batch_data).
        """
        n = min(self.batch_size, self.num_queries)
        if self.train_without_replacement:
            if self.num_queries <= 0:
                raise RuntimeError("train_without_replacement requires at least one train query")
            if self._train_perm is None or self._train_cursor >= self.num_queries:
                self._reshuffle_train_perm()
            start = self._train_cursor
            end = min(start + n, self.num_queries)
            indices = self._train_perm[start:end]
            self._train_cursor = end
            n = int(indices.numel())
        else:
            indices = torch.randint(0, self.num_queries, (n,))
        batch_queries = self.queries[indices]
        batch_edge_indices = self.query_edge_indices[indices % len(self.query_edge_indices)]

        if getattr(self.data, "has_inverse_edges", False):
            num_to_reverse = n // 2
            reverse_idx = torch.randperm(n)[:num_to_reverse]
            batch_queries[reverse_idx] = _inverse_queries(
                batch_queries[reverse_idx], self.data.num_relations
            )
        elif self.use_direction_head_queries:
            num_to_head = n // 2
            head_idx = torch.randperm(n)[:num_to_head]
            batch_queries[head_idx, 3] = 1

        batch_data = self._mask_batch_edges(batch_edge_indices)
        return batch_queries, batch_data

    def _mask_batch_edges(self, batch_edge_indices: Tensor) -> Data:
        """Remove batch edges and their inverses from visible graph for training.
        
        This prevents information leakage through inverse edges during training.
        
        Args:
            batch_edge_indices: Indices of edges in the current batch.
            
        Returns:
            Data object with batch edges and their inverses removed from the graph.
        """
        # Create mask: visible_mask AND NOT(batch edges)
        visible_mask = self.data.visible_mask.clone()
        visible_mask[batch_edge_indices] = False

        # Also mask the reverse of each batch edge to prevent leakage. This
        # applies both to KG graphs with explicit inverse relations
        # (``has_inverse_edges``) and to undirected single-relation LP graphs,
        # which carry an ``inverse_edge_map`` without doubling relations.
        inverse_edge_map = getattr(self.data, "inverse_edge_map", None)
        if inverse_edge_map is not None:
            inverse_indices = inverse_edge_map[batch_edge_indices]
            if getattr(self.data, "has_inverse_edges", False):
                assert (self.data.edge_type[batch_edge_indices] == (self.data.edge_type[inverse_indices] + self.data.num_relations // 2) % self.data.num_relations).all()
            visible_mask[inverse_indices] = False

        return self._create_masked_data(visible_mask)
    
    def _get_visible_data(self) -> Data:
        """Get data with all visible edges (for evaluation)."""
        return self._create_masked_data(self.data.visible_mask)
    
    def _create_masked_data(self, edge_mask: Tensor) -> Data:
        """Create a new Data object with only the masked edges.
        
        Args:
            edge_mask: Boolean mask indicating which edges to include.
            
        Returns:
            New Data object with filtered edges and recomputed CSR format.
        """
        # Filter edges
        edge_index = self.data.edge_index[:, edge_mask]
        edge_type = self.data.edge_type[edge_mask]

        new_data = Data(
            x=self.data.x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=self.data.num_nodes,
            num_relations=self.data.num_relations,
            y=self.data.y if hasattr(self.data, 'y') else None,
            edge_set=(
                self._train_edge_set
                if self._train_edge_set is not None
                else (self.data.edge_set if hasattr(self.data, "edge_set") else None)
            ),
        )
        
        # Compute CSR format (rowptr, indices) for the model
        if self.model_type == "wander":
            new_data = self._add_csr_format(new_data)
        
        return new_data
    
    def _add_csr_format(self, data: Data) -> Data:
        """Add CSR format (rowptr, indices, rel_indices) to the data object.
        
        Creates CSR format by sorting edges by source node. The edge_type is
        reordered to match the CSR order so that rel_indices[i] corresponds to
        the edge from rowptr to indices[i].
        
        Args:
            data: Data object with edge_index and edge_type.
            
        Returns:
            Data object with rowptr, indices, and rel_indices attributes.
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        edge_type = data.edge_type
        device = edge_index.device
        
        if edge_index.shape[1] == 0:
            # Handle empty graph
            data.rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
            data.indices = torch.tensor([], dtype=torch.long, device=device)
            data.rel_indices = torch.tensor([], dtype=torch.long, device=device)
            return data
        
        # Sort edges by source node to create CSR format
        # This ensures edge_type is reordered to match CSR indices
        src = edge_index[0]
        dst = edge_index[1]
        
        # Sort by source node (stable sort preserves order within same source)
        perm = torch.argsort(src, stable=True)
        sorted_src = src[perm]
        sorted_dst = dst[perm]
        sorted_edge_type = edge_type[perm]
        
        # Create rowptr by counting edges per source node
        # rowptr[i] = number of edges with src < i
        rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        # Count edges per source
        src_counts = torch.bincount(sorted_src, minlength=num_nodes)
        # Cumulative sum to get row pointers
        rowptr[1:] = torch.cumsum(src_counts, dim=0)
        
        # indices is just the sorted destination nodes
        indices = sorted_dst
        
        # rel_indices is the sorted edge types (matching CSR order)
        rel_indices = sorted_edge_type
        
        data.rowptr = rowptr
        data.indices = indices
        data.rel_indices = rel_indices
        
        return data


class BatchLoaderNodeClassification:
    """Custom loader that batches node classification queries as node index tensors.

    Training batches are class-balanced or uniform random (see ``balanced_training``).
    ``train_without_replacement`` is allowed only with uniform sampling: a
    shuffled deck of train nodes continues across epochs and reshuffles when empty.
    Evaluation batches shuffle nodes before subsampling and batching, sharded across DDP ranks.
    """

    def __init__(
        self,
        data: Data,
        mask: Tensor,
        batch_size: int = 8,
        training: bool = True,
        rank: int = 0,
        world_size: int = 1,
        max_eval_samples: Optional[int] = None,
        balanced_training: bool = True,
        eval_subsample_seed: Optional[int] = None,
        proximity_batching: bool = False,
        proximity_batch_max_radius: Optional[int] = None,
        proximity_batch_seed: Optional[int] = None,
        train_without_replacement: bool = False,
    ):
        self.data = data
        self.batch_size = batch_size
        self.training = training
        self.max_eval_samples = max_eval_samples
        self.balanced_training = balanced_training
        self.proximity_batching = bool(proximity_batching) and not training
        self.proximity_batch_max_radius = proximity_batch_max_radius
        self.proximity_batch_seed = proximity_batch_seed
        self._rank = rank
        self._world_size = world_size
        if training and train_without_replacement and balanced_training:
            raise ValueError(
                "train_without_replacement is not supported with class-balanced "
                "NC batches; use --node_cls_random_training_batches."
            )
        self.train_without_replacement = bool(train_without_replacement) and bool(training)
        self._train_perm: Optional[Tensor] = None
        self._train_cursor = 0
        self._reshuffle_count = 0
        self._eval_subsample_seed = eval_subsample_seed

        all_indices = mask.nonzero(as_tuple=True)[0]
        self._training_set_size = 1#len(all_indices) * 100

        assert data.y is not None and data.y.ndim == 2
        self.num_classes = data.y.shape[1]

        # Unsharded eval set: proximity batching clusters over the full mask
        # first, then shards whole batches across ranks (interleaved node
        # sharding would destroy locality).
        self._all_eval_indices = all_indices
        self._proximity_batches: Optional[List[Tensor]] = None
        self._proximity_n_global: Optional[int] = None
        self._ddp_eval_batches: Optional[List[Tensor]] = None
        self._ddp_eval_global_count: Optional[int] = None

        if world_size > 1 and not training:
            self.eval_node_indices = all_indices
        elif world_size > 1:
            self.eval_node_indices = all_indices[rank::world_size]
        else:
            self.eval_node_indices = all_indices

        self._eval_node_indices_fixed: Optional[Tensor] = None
        if (
            max_eval_samples is not None
            and eval_subsample_seed is not None
            and not (world_size > 1 and not training)
        ):
            self._eval_node_indices_fixed = _subsample_indices(
                self.eval_node_indices,
                max_eval_samples,
                subsample_seed=eval_subsample_seed,
            )

        if training:
            if self.train_without_replacement and world_size > 1:
                self.train_node_indices = all_indices[rank::world_size]
            else:
                self.train_node_indices = all_indices
            if balanced_training:
                labels = data.y[self.train_node_indices].argmax(dim=1)
                self.class_to_nodes: dict[int, Tensor] = {}
                for c in range(self.num_classes):
                    class_mask = (labels == c)
                    self.class_to_nodes[c] = self.train_node_indices[class_mask]

    def _rank_proximity_batches(self) -> List[Tensor]:
        """Build (lazily, once) and return this rank's BFS proximity batches.

        Batches are built deterministically over the full eval set, capped at
        whole batches until ``max_eval_samples * world_size`` nodes, then
        sharded round-robin (whole batches) across DDP ranks.
        Packed truncated balls around new seeds fill full-size batches
        (never continuing a frontier across a cut).
        """
        if self._proximity_batches is not None:
            return self._proximity_batches
        num_nodes = int(self.data.num_nodes)
        batches = _build_bfs_proximity_batches_new_seeds(
            self.data.edge_index,
            num_nodes,
            self._all_eval_indices,
            self.batch_size,
            max_radius=self.proximity_batch_max_radius,
            batch_seed=self.proximity_batch_seed,
        )
        if self.max_eval_samples is not None:
            cap = self.max_eval_samples * self._world_size
            selected: List[Tensor] = []
            total = 0
            for b in batches:
                if total >= cap:
                    break
                selected.append(b)
                total += int(b.numel())
            batches = selected
        self._proximity_n_global = sum(int(b.numel()) for b in batches)
        self._proximity_batches = _ddp_padded_batch_shard(
            batches, self._rank, self._world_size,
        )
        return self._proximity_batches

    def _build_ddp_eval_batches(self) -> List[Tensor]:
        """Non-proximity DDP eval: global subsample, chunk, padded batch shard."""
        if self._ddp_eval_batches is not None:
            return self._ddp_eval_batches
        global_indices = self._all_eval_indices
        if self.max_eval_samples is not None:
            cap = self.max_eval_samples * self._world_size
            global_indices = _subsample_indices(
                global_indices,
                cap,
                subsample_seed=self._eval_subsample_seed,
            )
        self._ddp_eval_global_count = int(global_indices.numel())
        batches = _chunk_indices(global_indices, self.batch_size)
        self._ddp_eval_batches = _ddp_padded_batch_shard(
            batches, self._rank, self._world_size,
        )
        return self._ddp_eval_batches

    def __len__(self) -> int:
        if self.training:
            return 1
        if not self.training and self._world_size > 1:
            if self.proximity_batching:
                return len(self._rank_proximity_batches())
            return len(self._build_ddp_eval_batches())
        if self.proximity_batching:
            return len(self._rank_proximity_batches())
        num_eval_nodes = len(self.eval_node_indices)
        return (num_eval_nodes + self.batch_size - 1) // self.batch_size

    def _get_batch_data(self) -> Data:
        return self.data

    def __iter__(self) -> Iterator[Tuple[Tensor, Data]]:
        if not self.training and self._world_size > 1:
            if self.proximity_batching:
                for chunk in self._rank_proximity_batches():
                    yield chunk, self._get_batch_data()
            else:
                for chunk in self._build_ddp_eval_batches():
                    yield chunk, self._get_batch_data()
            return
        if not self.training and self.proximity_batching:
            for chunk in self._rank_proximity_batches():
                yield chunk, self._get_batch_data()
            return
        if self.training:
            node_indices = self.train_node_indices
        elif self._eval_node_indices_fixed is not None:
            node_indices = self._eval_node_indices_fixed
        else:
            node_indices = _subsample_indices(
                self.eval_node_indices, self.max_eval_samples,
            )
        for start in range(0, len(node_indices), self.batch_size):
            chunk = node_indices[start:start + self.batch_size]
            yield chunk, self._get_batch_data()

    def eval_scoring_node_indices(self) -> Tensor:
        """Eval nodes used for batching and accuracy (matches ``__iter__`` in eval mode)."""
        if not self.training and self._world_size > 1:
            if self.proximity_batching:
                batches = self._rank_proximity_batches()
            else:
                batches = self._build_ddp_eval_batches()
            if not batches:
                return self._all_eval_indices[:0]
            nonempty = [b for b in batches if b.numel() > 0]
            if not nonempty:
                return self._all_eval_indices[:0]
            return torch.cat(nonempty)
        if self.proximity_batching:
            batches = self._rank_proximity_batches()
            if not batches:
                return self._all_eval_indices[:0]
            return torch.cat(batches)
        if self._eval_node_indices_fixed is not None:
            return self._eval_node_indices_fixed
        if self.max_eval_samples is not None:
            return _subsample_indices(self.eval_node_indices, self.max_eval_samples)
        return self.eval_node_indices

    def eval_scoring_node_count_global(self) -> int:
        """Unsharded eval node count (union of all ranks' batches)."""
        if not self.training and self._world_size > 1 and not self.proximity_batching:
            self._build_ddp_eval_batches()
            return int(self._ddp_eval_global_count or 0)
        if self.proximity_batching:
            self._rank_proximity_batches()
            return int(self._proximity_n_global or 0)
        if self._eval_node_indices_fixed is not None:
            return int(self._eval_node_indices_fixed.numel()) * max(int(self._world_size), 1)
        n = int(self._all_eval_indices.numel())
        if self.max_eval_samples is not None:
            return min(int(self.max_eval_samples) * max(int(self._world_size), 1), n)
        return n

    def _reshuffle_train_perm(self) -> None:
        """Draw a new no-replacement deck. Does not run at epoch boundaries."""
        n = int(self.train_node_indices.numel())
        if n <= 0:
            self._train_perm = torch.empty(0, dtype=torch.long)
            self._train_cursor = 0
            return
        g = torch.Generator()
        g.manual_seed(int(self._reshuffle_count) + 10007 * int(self._rank))
        self._train_perm = torch.randperm(n, generator=g)
        self._train_cursor = 0
        self._reshuffle_count += 1

    def set_epoch(self, epoch: int) -> None:
        """Ensure a no-replacement deck exists; do not reset the cursor."""
        if not self.train_without_replacement:
            return
        if self._train_perm is None:
            self._reshuffle_train_perm()

    def remaining_train_queries(self) -> int:
        """Unconsumed train nodes in the current no-replacement deck."""
        n = int(self.train_node_indices.numel())
        if not self.train_without_replacement:
            return n
        if self._train_perm is None:
            return n
        return max(0, n - int(self._train_cursor))

    def sample_batch(self) -> Tuple[Tensor, Data]:
        """Sample a training batch (class-balanced or uniform random)."""
        if not self.balanced_training:
            n_pool = int(self.train_node_indices.numel())
            n = min(self.batch_size, n_pool)
            if self.train_without_replacement:
                if n_pool <= 0:
                    raise RuntimeError("train_without_replacement requires at least one train node")
                if self._train_perm is None or self._train_cursor >= n_pool:
                    self._reshuffle_train_perm()
                start = self._train_cursor
                end = min(start + n, n_pool)
                selected = self.train_node_indices[self._train_perm[start:end]]
                self._train_cursor = end
            else:
                idx = torch.randint(0, n_pool, (n,))
                selected = self.train_node_indices[idx]
            selected = selected[torch.randperm(selected.shape[0])]
            return selected, self._get_batch_data()
        non_empty = [c for c in range(self.num_classes) if len(self.class_to_nodes[c]) > 0]
        k = len(non_empty)

        if self.batch_size <= k:
            chosen = [non_empty[i] for i in torch.randperm(k)[:self.batch_size].tolist()]
            selected = []
            for c in chosen:
                pool = self.class_to_nodes[c]
                selected.append(pool[torch.randint(0, len(pool), (1,))])
        else:
            per_class = self.batch_size // k
            remainder = self.batch_size - per_class * k
            selected = []
            for c in non_empty:
                pool = self.class_to_nodes[c]
                selected.append(pool[torch.randint(0, len(pool), (per_class,))])
            if remainder > 0:
                extra_classes = [non_empty[i] for i in torch.randperm(k)[:remainder].tolist()]
                for c in extra_classes:
                    pool = self.class_to_nodes[c]
                    selected.append(pool[torch.randint(0, len(pool), (1,))])

        selected = torch.cat(selected, dim=0)
        selected = selected[torch.randperm(selected.shape[0])]
        return selected, self._get_batch_data()


class QueryLoaderRecallAtK:
    """Node-based eval loader for the AnyGraph-style Recall@K protocol.

    Unlike :class:`QueryLoaderLinkPrediction` (one query per test edge), this
    loader yields one query per *source node* that appears in the split's
    forward query edges. Each batch row is ``(src, 0, 0)`` (tail prediction,
    placeholder tail) over the full visible graph; the model scores all nodes
    and the experiment evaluates Recall@K / NDCG@K against the source's set of
    true test tails, after masking train positives (and, for bipartite graphs,
    non-item candidates).

    Modeled on ``AnyGraph/data_handler.py`` ``TstData`` and ``AnyGraph/main.py``
    ``calc_recall_ndcg``. Evaluation only (no training path).
    """

    def __init__(
        self,
        data: Data,
        mask: Tensor,
        batch_size: int,
        eval_split: SplitName = "test",
        rank: int = 0,
        world_size: int = 1,
        model_type: str = "wander",
        max_eval_samples: Optional[int] = None,
        eval_subsample_seed: Optional[int] = None,
    ):
        self.data = data
        self.batch_size = max(1, batch_size)
        self.num_nodes = int(data.num_nodes)
        self.is_bipartite = bool(getattr(data, "anygraph_bipartite", False))
        self.candidate_offset = (
            int(getattr(data, "anygraph_candidate_offset", 0)) if self.is_bipartite else 0
        )
        self.eval_split = eval_split
        self.test_targets: Dict[int, List[int]] = {}
        # Restrict the query mask to forward edges (inverse edges duplicate flags).
        if getattr(data, "has_inverse_edges", False):
            n_forward = data.edge_index.shape[1] // 2
            fwd_mask = mask[:n_forward]
        else:
            fwd_mask = mask
        idx = fwd_mask.nonzero(as_tuple=True)[0]
        heads = data.edge_index[0, idx].tolist()
        tails = data.edge_index[1, idx].tolist()
        for h, t in zip(heads, tails):
            self.test_targets.setdefault(h, []).append(t)

        sources = torch.tensor(sorted(self.test_targets.keys()), dtype=torch.long)
        if world_size > 1:
            sources = sources[rank::world_size]
        if max_eval_samples is not None:
            sources = _subsample_indices(sources, max_eval_samples, eval_subsample_seed)
        self.sources = sources

        # Build the (cached) visible eval graph by reusing the LP loader machinery.
        self._inner = QueryLoaderLinkPrediction(
            data=data,
            mask=mask,
            batch_size=self.batch_size,
            shuffle=False,
            training=False,
            rank=rank,
            world_size=world_size,
            model_type=model_type,
            max_eval_samples=None,
        )
        self._visible_data = self._inner._get_visible_data()

    def __len__(self) -> int:
        return (len(self.sources) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, Data]]:
        for start in range(0, len(self.sources), self.batch_size):
            src = self.sources[start:start + self.batch_size]
            zeros = torch.zeros_like(src)
            batch_queries = torch.stack([src, zeros, zeros], dim=1)
            yield batch_queries, self._visible_data


class QueryLoaderSampledRecall:
    """One query per source on a cached sampled-recall candidate list.

    Yields ``(src, 0, 0)`` plus the visible train graph. The experiment scores
    ``pos ∪ sampled-neg`` via ``candidate_tails`` (see ``_iter_candidates``).
    """

    def __init__(
        self,
        data: Data,
        eval_split: SplitName = "test",
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        model_type: str = "wander",
        mask: Optional[Tensor] = None,
        num_neg: int = DEFAULT_NUM_NEG,
        max_sources: Optional[int] = None,
    ):
        self.data = data
        self.batch_size = max(1, batch_size)
        self.eval_split = eval_split
        self._iter_candidates: Optional[List[Tuple[int, Tensor, Tensor]]] = None

        split_key = "valid" if eval_split == "val" else eval_split
        if split_key == "test":
            cache_attr = "sampled_recall_test"
        elif split_key in ("val", "valid"):
            cache_attr = "sampled_recall_valid"
        else:
            raise ValueError(f"sampled recall eval split must be val/test, got {eval_split!r}")

        cache = getattr(data, cache_attr, None)
        if cache is None:
            split_edge = getattr(data, "unilp_split_edge", None)
            data_dir = getattr(data, "unilp_data_dir", None)
            folder = getattr(data, "unilp_folder", None)
            run = int(getattr(data, "unilp_run", 0))
            if split_edge is None or data_dir is None or folder is None:
                raise AttributeError(
                    "sampled recall requires unilp_split_edge / unilp_data_dir "
                    f"on the graph (missing cache attr {cache_attr})"
                )
            cache_dir = Path(data_dir) / folder
            cache = load_or_build_sampled_recall_cache(
                cache_dir,
                split_edge,
                int(data.num_nodes),
                run=run,
                split=split_key,
                num_neg=int(num_neg),
            )
            setattr(data, cache_attr, cache)
        if max_sources is not None and int(max_sources) > 0:
            from data.sampled_recall import subsample_sampled_recall_cache

            cache = subsample_sampled_recall_cache(cache, int(max_sources))
        self.cache = cache
        n_src = int(cache["sources"].numel())
        indices = torch.arange(n_src, dtype=torch.long)
        if world_size > 1:
            indices = indices[rank::world_size]
        self.source_indices = indices
        self.num_queries = int(indices.numel())

        dummy_mask = mask
        if dummy_mask is None:
            dummy_mask = getattr(data, "train_mask", None)
        if dummy_mask is None:
            dummy_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
        self._inner = QueryLoaderLinkPrediction(
            data=data,
            mask=dummy_mask,
            batch_size=self.batch_size,
            shuffle=False,
            training=False,
            rank=0,
            world_size=1,
            model_type=model_type,
            max_eval_samples=None,
        )
        self._visible_data = self._inner._get_visible_data()

    def __len__(self) -> int:
        if self.num_queries == 0:
            return 0
        return (self.num_queries + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, Data]]:
        from data.sampled_recall import iter_source_candidates

        for start in range(0, self.num_queries, self.batch_size):
            sl = self.source_indices[start : start + self.batch_size]
            rows = iter_source_candidates(self.cache, sl.tolist())
            self._iter_candidates = rows
            src = torch.tensor([r[0] for r in rows], dtype=torch.long)
            zeros = torch.zeros_like(src)
            yield torch.stack([src, zeros, zeros], dim=1), self._visible_data


class MultiGraphLoader:
    """Loader that handles multiple graphs for multi-graph training/evaluation."""

    def __init__(
        self,
        bundles: Dict[str, GraphBundle],
        split: SplitName,
        batch_size_node: int = 1,
        batch_size_link: int = 1,
        shuffle: bool = True,
        training: bool = True,
        rank: int = 0,
        world_size: int = 1,
        batch_per_epoch: int = 100,
        model_type: str = "wander",
        max_eval_samples: Optional[int] = None,
        balanced_training: bool = True,
        num_graphs_per_step: int = 1,
        use_direction_head_queries: bool = False,
        evaluate_head_predictions: Optional[bool] = None,
        eval_subsample_seed: Optional[int] = None,
        link_pred_eval: Optional[str] = None,
        recall_k: int = 20,
        sampled_recall_num_neg: int = DEFAULT_NUM_NEG,
        sampled_recall_max_sources: Optional[int] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        nc_proximity_batching: bool = False,
        nc_proximity_batch_max_radius: Optional[int] = None,
        nc_proximity_batch_seed: Optional[int] = None,
        train_without_replacement: bool = False,
    ):
        self.bundles = bundles
        self.split = split
        self.link_pred_eval = link_pred_eval
        self.recall_k = recall_k
        self.sampled_recall_num_neg = int(sampled_recall_num_neg)
        self.sampled_recall_max_sources = (
            int(sampled_recall_max_sources) if sampled_recall_max_sources else None
        )
        self.datasets: Dict[str, Data] = {}
        self.dataset_eval_splits: Dict[str, SplitName] = {}
        self.training = training
        self.dataset_names = list(bundles.keys())
        self._epoch = 0
        self._world_size = world_size
        self._rank = rank
        self.batch_per_epoch = batch_per_epoch
        self.num_graphs_per_step = max(1, num_graphs_per_step)
        self.use_direction_head_queries = use_direction_head_queries
        self.evaluate_head_predictions = evaluate_head_predictions
        self._dataset_weights = dataset_weights or {}
        self.train_without_replacement = bool(train_without_replacement) and bool(training)

        self.loaders = {}
        for name, bundle in bundles.items():
            eff_split: SplitName = split
            if not training and split == "val":
                pref = bundle.metadata.get("preferred_eval_split")
                if pref in ("train", "val", "test"):
                    eff_split = pref  # type: ignore[assignment]
            self.dataset_eval_splits[name] = eff_split
            data = resolve_split(bundle, eff_split)
            self.datasets[name] = data
            train_data = resolve_split(bundle, "train")
            train_edge_set = getattr(train_data, "edge_set", None)
            ds_eval_seed = (
                _dataset_eval_subsample_seed(eval_subsample_seed, name)
                if eval_subsample_seed is not None
                else None
            )
            if is_link_prediction(data):
                mask = query_mask_for_split(data, eff_split)
                # Resolve the eval protocol per dataset: explicit CLI value wins,
                # else the dataset's preferred setting (bundle.metadata), else the
                # global default (mrr / no head predictions).
                eff_eval = link_pred_eval
                if eff_eval is None:
                    eff_eval = bundle.metadata.get("preferred_link_pred_eval")
                if eff_eval is None:
                    eff_eval = "mrr"
                eff_head = evaluate_head_predictions
                if eff_head is None:
                    eff_head = bundle.metadata.get("preferred_evaluate_head_predictions")
                if eff_head is None:
                    eff_head = False
                if eff_eval == "recall" and not training:
                    self.loaders[name] = QueryLoaderRecallAtK(
                        data=data,
                        mask=mask,
                        batch_size=batch_size_link,
                        eval_split=eff_split,
                        rank=rank,
                        world_size=world_size,
                        model_type=model_type,
                        max_eval_samples=max_eval_samples,
                        eval_subsample_seed=ds_eval_seed,
                    )
                    continue
                if eff_eval == "sampled_recall" and not training:
                    self.loaders[name] = QueryLoaderSampledRecall(
                        data=data,
                        eval_split=eff_split,
                        batch_size=batch_size_link,
                        rank=rank,
                        world_size=world_size,
                        model_type=model_type,
                        mask=mask,
                        num_neg=self.sampled_recall_num_neg,
                        max_sources=self.sampled_recall_max_sources,
                    )
                    continue
                if eff_eval == "hits":
                    raise ValueError(
                        "Pooled UniLP Hits@K eval was removed; use "
                        "--link_pred_eval sampled_recall"
                    )
                self.loaders[name] = QueryLoaderLinkPrediction(
                    data=data,
                    mask=mask,
                    batch_size=batch_size_link,
                    shuffle=shuffle,
                    training=training,
                    rank=rank,
                    world_size=world_size,
                    model_type=model_type,
                    max_eval_samples=max_eval_samples,
                    train_edge_set=train_edge_set if training else None,
                    use_direction_head_queries=use_direction_head_queries,
                    evaluate_head_predictions=(
                        bool(eff_head) if not training else False
                    ),
                    eval_subsample_seed=ds_eval_seed,
                    train_without_replacement=self.train_without_replacement,
                )
            else:
                mask = getattr(data, f"{split}_mask")
                self.loaders[name] = BatchLoaderNodeClassification(
                    data=data,
                    mask=mask,
                    batch_size=batch_size_node,
                    training=training,
                    rank=rank,
                    world_size=world_size,
                    max_eval_samples=max_eval_samples,
                    balanced_training=balanced_training,
                    eval_subsample_seed=ds_eval_seed,
                    proximity_batching=nc_proximity_batching,
                    proximity_batch_max_radius=nc_proximity_batch_max_radius,
                    proximity_batch_seed=nc_proximity_batch_seed,
                    train_without_replacement=self.train_without_replacement,
                )

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic dataset ordering across DDP ranks."""
        self._epoch = epoch
        for loader in self.loaders.values():
            set_ep = getattr(loader, "set_epoch", None)
            if callable(set_ep):
                set_ep(epoch)

    def __len__(self) -> int:
        """Total number of batches across all graphs."""
        if self.training:
            return self.batch_per_epoch
        return sum(len(loader) for loader in self.loaders.values())
    
    def __iter__(self) -> Iterator[LoaderYield]:
        """Iterate over batches.

        Training: yields one item per optimizer step — a singleton when this rank
        runs a single graph in the step, else ``list[(name, batch, data)]`` (DDP:
        length ``num_graphs_per_step // world_size`` when ``world_size > 1``).
        Evaluation: always yields ``(name, batch, data)`` singletons.
        """
        if self.training:
            yield from self._training_iter()
        else:
            yield from self._eval_iter()

    def _training_iter(self) -> Iterator[LoaderYield]:
        """Weighted random sampling across graphs for training.

        Produces exactly ``self.batch_per_epoch`` yields (= optimizer steps per rank).
        Each global step samples ``num_graphs_per_step`` graphs using a generator
        seeded from ``self._epoch`` so all DDP ranks share the same sequence of
        choices; each rank processes a contiguous slice of size
        ``num_graphs_per_step // world_size`` (single-GPU: full ``num_graphs_per_step``).
        At each sampled graph, queries are drawn with probability proportional to
        training-mask size for dataset selection, then ``sample_batch`` samples
        from that graph (with replacement unless ``train_without_replacement``).
        """

        weights = torch.tensor(
            [
                self.loaders[name]._training_set_size
                * self._dataset_weights.get(name, 1.0)
                for name in self.dataset_names
            ],
            dtype=torch.float,
        )
        if weights.sum() <= 0:
            raise ValueError(
                "MultiGraphLoader dataset sampling weights sum to zero."
            )
        weights = weights / weights.sum()

        g = torch.Generator()
        g.manual_seed(self._epoch)

        ws = self._world_size
        ng = self.num_graphs_per_step
        per_rank = ng // ws if ws > 1 else ng
        log_real = os.environ.get("LOG_REAL_TRAIN_GRAPHS", "0") == "1"

        def _sample(name: str) -> StepItem:
            batch, data = self.loaders[name].sample_batch()
            return name, batch, data

        def _log_real_step(step_idx: int, names: list[str]) -> None:
            if not log_real:
                return
            print(
                f"[real_train] epoch={self._epoch} step={step_idx} "
                f"rank={self._rank} datasets={names}",
                flush=True,
            )

        for step_idx in range(self.batch_per_epoch):
            if ws == 1:
                if ng == 1:
                    graph_idx = torch.multinomial(weights, 1, generator=g).item()
                    name = self.dataset_names[graph_idx]
                    _log_real_step(step_idx, [name])
                    yield _sample(name)
                else:
                    group: List[StepItem] = []
                    names: list[str] = []
                    for _g in range(ng):
                        graph_idx = torch.multinomial(weights, 1, generator=g).item()
                        name = self.dataset_names[graph_idx]
                        names.append(name)
                        group.append(_sample(name))
                    _log_real_step(step_idx, names)
                    yield group
                continue

            # Shared multinomial sequence across ranks (same generator seed);
            # only materialize this rank's slice.
            names = []
            for _g in range(ng):
                graph_idx = torch.multinomial(weights, 1, generator=g).item()
                names.append(self.dataset_names[graph_idx])

            start = self._rank * per_rank
            my_names = names[start : start + per_rank]
            _log_real_step(step_idx, my_names)
            sub = [_sample(name) for name in my_names]
            if len(sub) == 1:
                yield sub[0]
            else:
                yield sub

    def _eval_iter(self) -> Iterator[StepItem]:
        """Sequential iteration - complete each graph before moving to next."""
        for name, loader in self.loaders.items():
            for batch, data in loader:
                yield (name, batch, data)

