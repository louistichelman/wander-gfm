"""Graph topology samplers used by ``SCMPrior``.

Covers graph-first generators (ER, SBM, Watts–Strogatz, geometric, …),
features-first data-conditioned graphs, and bipartite recommendation graphs.
"""

import abc
import random
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def _make_graph(
    edge_index: Tensor,
    num_nodes: int,
    device: torch.device,
    **node_attrs: Tensor,
) -> Data:
    """Build a PyG ``Data`` with explicit ``num_nodes`` and optional node stores."""
    g = Data(
        edge_index=edge_index.to(device),
        num_nodes=num_nodes,
    )
    for k, v in node_attrs.items():
        g[k] = v.to(device)
    return g


def _graph_device(graph: Data) -> torch.device:
    return graph.edge_index.device


def _to_scipy_csr(graph: Data):
    """Convert a PyG Data to SciPy CSR adjacency (for connected_components)."""
    ei = graph.edge_index.cpu()
    n = graph.num_nodes
    ones = np.ones(ei.shape[1], dtype=np.int32)
    return csr_matrix((ones, (ei[0].numpy(), ei[1].numpy())), shape=(n, n))


class GraphSamplerBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        n_nodes: int,
        device: torch.device,
        strict_n_nodes: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.strict_n_nodes = strict_n_nodes

    @abc.abstractmethod
    def forward(self) -> Data:
        raise NotImplementedError()

    def sample(self) -> Data:
        graph = self.forward()
        if self.strict_n_nodes:
            assert graph.num_nodes == self.n_nodes
        assert _graph_device(graph) == self.device
        return graph


# >>> Utils


def shuffle_graph(graph: Data) -> Data:
    n_nodes = graph.num_nodes
    device = _graph_device(graph)

    perm = torch.randperm(n_nodes, device=device)
    perm_inv = torch.argsort(perm)

    src, dst = graph.edge_index
    src, dst = perm[src], perm[dst]

    new_graph = _make_graph(
        torch.stack([src, dst]),
        num_nodes=n_nodes,
        device=device,
    )
    return new_graph


def create_graph_from_edgelist(
    src: Tensor,
    dst: Tensor,
    *,
    device: torch.device | str,
    add_reverse_edges: bool = True,
    n_nodes: int | None = None,
) -> Data:
    if n_nodes is None:
        n_nodes = int(max(src.max().item(), dst.max().item())) + 1

    if add_reverse_edges:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])

    edge_index = torch.stack([src, dst]).to(device)
    return _make_graph(edge_index, num_nodes=n_nodes, device=torch.device(device))


def graph_from_adj(
    adj: Tensor,
    *,
    device: torch.device,
    undirected: bool = True,
) -> Data:
    n_nodes = adj.shape[0]
    if undirected:
        adj = adj.triu()
        adj = torch.where(
            torch.eye(n_nodes, device=adj.device, dtype=torch.bool),
            0,
            adj,
        )
    src, dst = torch.nonzero(adj, as_tuple=True)
    return create_graph_from_edgelist(
        src,
        dst,
        device=device,
        add_reverse_edges=True,
        n_nodes=n_nodes,
    )


def adjust_intensity(
    intensity: Tensor,
    max_binsearch_iters: int = 1_000,
    rtol: float = 1e-4,
) -> Tensor:
    intensity = intensity.double()
    initial_sum = intensity.sum()
    f = lambda c: (c * intensity).clip(max=1.0).sum().item()  # noqa: E731
    c_lb = 1.0
    c_rb = 1 / intensity[intensity > 0].min()

    if (intensity > 0).float().sum() < (1 + rtol) * initial_sum:
        intensity = (c_rb * intensity).clip(max=1)
        return intensity

    assert f(c_lb) <= (1 + rtol) * initial_sum
    assert f(c_rb) >= (1 - rtol) * initial_sum, (
        f(c_rb),
        c_rb,
        intensity[intensity > 0].min(),
        initial_sum,
        (intensity > 0).float().sum(),
    )

    for _ in range(max_binsearch_iters):
        c_mid = (c_lb + c_rb) / 2
        f_value = f(c_mid)
        if abs(f_value - initial_sum) < rtol * initial_sum:
            break
        elif f_value < initial_sum:
            c_lb = c_mid
        else:
            c_rb = c_mid

    c = (c_lb + c_rb) / 2
    intensity = (c * intensity).clip(max=1)
    return intensity


# TODO: rewrite to binary search
def sample_from_intensity(
    intensity: Tensor,
) -> Tensor:
    intensity = adjust_intensity(intensity).clip(max=1)
    adj = (torch.rand_like(intensity) < intensity).to(torch.int32)
    return adj


def random_partition(
    n: int,
    n_parts: int,
    *,
    min_size: int = 1,
    min_part_ratio: float | None = 0.1,  # TODO: maybe rename
) -> list[int]:
    """
    Generates a random partition: list of positive integers of len n_parts summing to n
    """
    if n_parts < 1 or n < n_parts:
        raise ValueError("Require 1 <= n_parts <= n.")
    if n_parts == 1:
        return [n]

    if min_part_ratio is not None:
        min_size = max(
            min_size,
            int((n / n_parts) * min_part_ratio),
        )

    assert min_size * n_parts <= n
    n -= (min_size - 1) * n_parts

    cuts = np.sort(np.random.choice(np.arange(1, n), size=n_parts - 1, replace=False))
    parts = np.diff(np.concatenate(([0], cuts, [n]))).astype(np.int32)
    parts = (min_size - 1) + parts
    return parts.tolist()  # type: ignore


def random_float(low: float, high: float) -> float:
    return np.random.random() * (high - low) + low


def random_bool(p: float = 0.5) -> bool:
    return np.random.random() < p


# <<<


# >>> Base Graph Samplers


class LineSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, device: torch.device):
        super().__init__(n_nodes, device)

    def forward(self) -> Data:
        nodes = torch.randperm(self.n_nodes, device=self.device)
        src, dst = nodes[:-1], nodes[1:]
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])]
        )
        return _make_graph(edge_index, num_nodes=self.n_nodes, device=self.device)


# TODO: maybe optimize
# TODO: rewrite to Prüfer sequence
class TreeSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, device: torch.device):
        super().__init__(n_nodes, device)

    def forward(self) -> Data:
        src = torch.arange(1, self.n_nodes, device=self.device)
        dst = []

        for i in range(1, self.n_nodes):
            dst.append(np.random.randint(0, i))

        dst = torch.tensor(dst, device=self.device)

        graph = create_graph_from_edgelist(
            src,
            dst,
            device=self.device,
            n_nodes=self.n_nodes,
        )
        graph = shuffle_graph(graph)
        return graph


class ERSampler(GraphSamplerBase):
    def __init__(self, n_nodes: int, avg_degree: float, device: torch.device):
        super().__init__(n_nodes, device)
        self.n_edges = int(n_nodes * avg_degree)

    def forward(self) -> Data:
        src = torch.randint(0, self.n_nodes, (self.n_edges,), device=self.device)
        dst = torch.randint(0, self.n_nodes, (self.n_edges,), device=self.device)
        edge_index = torch.stack([src, dst])
        return _make_graph(edge_index, num_nodes=self.n_nodes, device=self.device)


class WattsStrogatzSampler(GraphSamplerBase):
    """Watts-Strogatz small-world graph: high clustering coefficient with short path lengths."""

    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        beta_min: float = 0.01,
        beta_max: float = 0.3,
        device: torch.device,
    ):
        super().__init__(n_nodes, device)
        self.k = max(2, int(avg_degree))
        if self.k % 2 != 0:
            self.k += 1
        self.beta = random_float(beta_min, beta_max)

    def forward(self) -> Data:
        import networkx as nx

        G = nx.watts_strogatz_graph(self.n_nodes, self.k, self.beta)
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            return _make_graph(edge_index, num_nodes=self.n_nodes, device=self.device)
        src, dst = zip(*edges)
        src = torch.tensor(src, device=self.device)
        dst = torch.tensor(dst, device=self.device)
        return create_graph_from_edgelist(
            src, dst, device=self.device, n_nodes=self.n_nodes
        )


# TODO: double-check and refactor
class DegreeCorrectedSBMSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        min_n_groups: int = 2,
        max_n_groups: int = 8,
        p_uniform_density: float = 0.5,
        p_power_law_degrees: float = 0.5,
        avg_degree: float,
        device: torch.device,
        offdiagonal_coef: float = 0.1,
    ):
        super().__init__(n_nodes, device)
        self.avg_degree = avg_degree
        self.offdiagonal_coef = offdiagonal_coef
        self.min_n_groups = min_n_groups
        self.max_n_groups = max_n_groups
        self.p_uniform_density = p_uniform_density
        self.p_power_law_degrees = p_power_law_degrees

    def get_groups(self, group_sizes: list[int]) -> Tensor:
        n_groups = len(group_sizes)
        groups = torch.cat(
            [
                torch.full(
                    [group_size], group_idx, dtype=torch.int32, device=self.device
                )
                for group_idx, group_size in enumerate(group_sizes)
            ],
            dim=0,
        )

        for i, group_size in enumerate(group_sizes):
            group_start = sum(group_sizes[:i])
            group_end = group_start + group_size
            groups[group_start:group_end] = i

        assert groups.ndim == 1
        assert groups.shape[0] == self.n_nodes
        assert groups.min().item() == 0
        assert groups.max().item() == n_groups - 1
        assert len(torch.unique(groups)) == n_groups
        return groups

    # TODO: rename
    def get_n_edges_expected(self, group_sizes: list[int]) -> Tensor:
        n_groups = len(group_sizes)
        uniform_density = random_bool(p=self.p_uniform_density)

        # >>> Generate n_edges_expected
        n_edges_expected = torch.zeros(
            [n_groups, n_groups],
            dtype=torch.float32,
            device=self.device,
        )

        # Generate diagonal values
        for i in range(n_groups):
            n_edges_expected[i, i] = 1.0 if uniform_density else np.random.random()

        # Generate off-diagonal values
        uniform_offdiagonal_density = np.random.random()
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                max_prob = (n_edges_expected[i, i] * n_edges_expected[j, j]) ** 0.5
                n_edges_expected[i, j] = self.offdiagonal_coef * (
                    uniform_offdiagonal_density
                    if uniform_density
                    else np.random.random() * max_prob
                )

        # >>> Normalize & return
        group_sizes_tensor = torch.tensor(group_sizes, device=self.device)
        n_edges_expected = (
            n_edges_expected * group_sizes_tensor[:, None] * group_sizes_tensor[None, :]
        )
        n_edges_expected = (
            n_edges_expected + n_edges_expected.T - torch.diag(n_edges_expected.diag())
        )

        return n_edges_expected

    def get_degs(self, group_sizes: list[int]) -> Tensor:
        # >>> Generate degs
        if random_bool(self.p_power_law_degrees):
            gamma = random_float(2, 3)
            degs = torch.tensor(
                np.random.zipf(gamma, [self.n_nodes]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            degs = torch.rand([self.n_nodes], dtype=torch.float32, device=self.device)

        # >>> Normalize
        for i, group_size in enumerate(group_sizes):
            group_start = sum(group_sizes[:i])
            group_end = group_start + group_size
            degs[group_start:group_end] /= degs[group_start:group_end].sum()

        # >>> Sanity checks & return
        assert torch.all(degs >= 0)
        return degs

    def sample_adjacency_matrix(
        self,
        groups: Tensor,
        n_edges_expected: Tensor,
        degs: Tensor,
    ) -> Tensor:
        intensity = (
            n_edges_expected[groups[:, None], groups[None, :]]
            * degs[:, None]
            * degs[None, :]
        ).triu()
        intensity = intensity / intensity.sum()
        intensity = (0.5 * self.n_nodes * self.avg_degree) * intensity

        # Sanity checks
        avg_edges = intensity.sum().item()
        avg_degree = 2 * avg_edges / self.n_nodes
        assert torch.allclose(
            torch.tensor(avg_degree),
            torch.tensor(self.avg_degree, dtype=torch.float32),
        )
        assert torch.allclose(intensity.triu(), intensity)
        assert torch.all(intensity >= 0)

        # Sample & return
        adj = sample_from_intensity(intensity)
        return adj

    def forward(self) -> Data:
        n_groups = np.random.randint(self.min_n_groups, self.max_n_groups)
        group_sizes = random_partition(self.n_nodes, n_groups)
        groups = self.get_groups(group_sizes)
        n_edges_expected = self.get_n_edges_expected(group_sizes)
        degs = self.get_degs(group_sizes)
        adj = self.sample_adjacency_matrix(groups, n_edges_expected, degs)
        graph = graph_from_adj(adj, device=self.device)
        graph = shuffle_graph(graph)
        return graph


# TODO: maybe optimize
class PreferentialAttachmentSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        device: torch.device,
        *,
        max_degree: int | None = None,
        base: GraphSamplerBase | None = None,
        avg_degree: float | None = None,
    ):
        assert (max_degree is None) ^ (avg_degree is None)
        if max_degree is None:
            assert avg_degree is not None
            max_degree = int(2 * avg_degree - 1)
        if base is None:
            self.n_nodes_pa = n_nodes
            strict_n_nodes = True
        else:
            assert base.n_nodes <= n_nodes
            self.n_nodes_pa = n_nodes - base.n_nodes
            strict_n_nodes = base.strict_n_nodes

        super().__init__(n_nodes, device, strict_n_nodes)
        self.base = base
        self.max_degree = max_degree

    def forward(self) -> Data:
        n_nodes = self.n_nodes_pa

        if self.base is not None:
            base_graph = self.base.sample()
            base_src, base_dst = base_graph.edge_index
            src, dst = base_src.tolist(), base_dst.tolist()
            from torch_geometric.utils import degree as pyg_degree
            degrees = pyg_degree(base_graph.edge_index[1], num_nodes=base_graph.num_nodes).long().tolist()
            n_nodes += base_graph.num_nodes
        else:
            src, dst = [0], [1]
            degrees = [1, 1]

        for _ in range(len(degrees), n_nodes):
            u = len(degrees)
            u_degree = random.randint(1, self.max_degree)
            u_degree = min(u_degree, len(degrees))
            p = np.array(degrees)
            p = p / p.sum()
            v_list = np.random.choice(len(degrees), [u_degree], replace=False, p=p)
            for v in v_list:
                src.append(u)
                src.append(v)
                dst.append(v)
                dst.append(u)
                degrees[v] += 1
            degrees.append(u_degree)

        graph = create_graph_from_edgelist(
            torch.tensor(src, device=self.device),
            torch.tensor(dst, device=self.device),
            device=self.device,
            n_nodes=n_nodes,
            add_reverse_edges=False,
        )
        graph = shuffle_graph(graph)
        return graph


class GeometricGraphSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        min_n_latent_features: int = 2,
        max_n_latent_features: int = 10,
        device: torch.device,
    ):
        super().__init__(n_nodes, device)

        self.avg_degree = avg_degree
        self.min_n_latent_features = min_n_latent_features
        self.max_n_latent_features = max_n_latent_features

    def generate_points(self) -> Tensor:
        n_latent_features = np.random.randint(
            self.min_n_latent_features,
            self.max_n_latent_features + 1,
        )
        if random_bool(p=0.5):
            random_generator = torch.rand
        else:
            random_generator = torch.randn
        points = random_generator((self.n_nodes, n_latent_features), device=self.device)
        return points

    def compute_distances(self, points: Tensor, how: Literal["l2"] = "l2") -> Tensor:
        if how == "l2":
            # [None, ...] and [0, ...] are here since cdist assumes batched input
            distances = torch.cdist(points[None, ...], points[None, ...])[0, ...]
        else:
            raise ValueError(f"Unknown type: {how}")
        return distances

    def compute_threshold(
        self,
        distances: Tensor,
        subsample_size: int = 100_000,
    ) -> float:
        n_edges = 0.5 * (self.n_nodes * self.avg_degree)
        distances = distances.triu()
        distances = distances[distances > 0.0]
        assert distances.ndim == 1
        q = n_edges / distances.shape[0]
        if distances.shape[0] > subsample_size:
            perm = torch.randperm(distances.shape[0], device=distances.device)
            distances = distances[perm]
            distances = distances[:subsample_size]
        return torch.quantile(distances, q).item()

    def forward(self) -> Data:
        points = self.generate_points()
        distances = self.compute_distances(points)
        threshold = self.compute_threshold(distances)
        adj = (distances < threshold).to(torch.int32)
        graph = graph_from_adj(adj, device=self.device)
        graph.features = points
        return graph


# <<<


# >>> Util Samplers & Postprocessing


# TODO: refactor everything to functional style
def _add_random_edges_connectivity(graph: Data) -> Data:
    """Add random undirected edges (both directions) only between the **largest**
    connected component and **other** components.

    Each new link connects one node sampled uniformly from the LCC and one node
    sampled uniformly from a component drawn uniformly among the non-LCC
    components.  Inserts ``max(1, int(0.05 * num_edges))`` new **directed**
    edges, in pairs ``(u,v)`` and ``(v,u)``, at least one pair.  If the graph is
    already connected, returns it unchanged.
    """
    n = graph.num_nodes
    m = int(graph.edge_index.size(1))
    if n < 2:
        return graph
    device = _graph_device(graph)
    adj_scipy = _to_scipy_csr(graph)
    n_comp, labels = connected_components(adj_scipy, directed=False)
    labels_t = torch.tensor(labels, device=device, dtype=torch.long)
    if n_comp <= 1:
        return graph

    counts = torch.bincount(labels_t, minlength=n_comp)
    largest_c = int(counts.argmax().item())
    non_largest = torch.tensor(
        [c for c in range(n_comp) if c != largest_c],
        device=device,
        dtype=torch.long,
    )
    if non_largest.numel() == 0:
        return graph

    n_new_directed = max(1, int(0.05 * m))
    n_pairs = max(1, (n_new_directed + 1) // 2)

    idx_other = torch.randint(0, non_largest.numel(), (n_pairs,), device=device)
    c_other = non_largest[idx_other]

    nodes_lcc = (labels_t == largest_c).nonzero(as_tuple=True)[0]
    u = torch.empty(n_pairs, dtype=torch.int64, device=device)
    v = torch.empty(n_pairs, dtype=torch.int64, device=device)
    for i in range(n_pairs):
        co = int(c_other[i].item())
        nodes_o = (labels_t == co).nonzero(as_tuple=True)[0]
        u[i] = nodes_lcc[torch.randint(0, nodes_lcc.numel(), (1,), device=device)]
        v[i] = nodes_o[torch.randint(0, nodes_o.numel(), (1,), device=device)]

    new_src = torch.cat([u, v])
    new_dst = torch.cat([v, u])
    new_edges = torch.stack([new_src, new_dst])
    edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
    return _make_graph(edge_index, num_nodes=n, device=device)


def extract_largest_component(graph: Data) -> tuple[Data, Tensor]:
    """Return the largest connected component as an induced subgraph.

    If the largest component has at most ``num_nodes // 2`` nodes, up to three
    rounds of random edge insertions (~5% of current edges per round) are
    applied to try to merge components before extracting the LCC.
    """
    n = graph.num_nodes
    device = _graph_device(graph)
    nodes = torch.arange(n, device=device)

    for _ in range(5):
        adj_scipy = _to_scipy_csr(graph)
        n_components, labels = connected_components(adj_scipy, directed=False)
        labels_t = torch.tensor(labels, device=device)
        if n_components == 1:
            break
        counts = torch.bincount(labels_t)
        lcc_size = int(counts.max().item())
        if lcc_size > n // 2:
            break
        num_isolated_nodes = (counts == 1).sum().item()
        print(f"num_isolated_nodes: {num_isolated_nodes}")
        graph = _add_random_edges_connectivity(graph)

    adj_scipy = _to_scipy_csr(graph)
    n_components, labels = connected_components(adj_scipy, directed=False)

    if n_components == 1:
        return graph, nodes

    labels = torch.tensor(labels, device=device)
    largest_component_idx = torch.bincount(labels).argmax()
    component_nodes = nodes[labels == largest_component_idx]

    # Build induced subgraph with relabelled nodes
    node_mask = labels == largest_component_idx
    node_map = torch.full((n,), -1, dtype=torch.long, device=device)
    node_map[component_nodes] = torch.arange(component_nodes.size(0), device=device)

    src, dst = graph.edge_index
    edge_mask = node_mask[src] & node_mask[dst]
    new_src = node_map[src[edge_mask]]
    new_dst = node_map[dst[edge_mask]]
    new_edge_index = torch.stack([new_src, new_dst])

    new_graph = _make_graph(new_edge_index, num_nodes=int(component_nodes.size(0)), device=device)
    return new_graph, component_nodes


class ExtractLargestComponent(GraphSamplerBase):
    def __init__(self, base: GraphSamplerBase):
        super().__init__(base.n_nodes, base.device, strict_n_nodes=False)
        self.base = base

    def forward(self) -> Data:
        graph = self.base.sample()
        graph, _ = extract_largest_component(graph)
        return graph


def _to_simple(graph: Data, device: torch.device) -> Data:
    """Bidirected + deduplicate + remove self-loops (mirrors old ToSimple)."""
    ei = graph.edge_index.cpu()
    ei = to_undirected(ei, num_nodes=graph.num_nodes)
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, num_nodes=graph.num_nodes)
    return _make_graph(ei.to(device), num_nodes=graph.num_nodes, device=device)


class ToSimple(GraphSamplerBase):
    def __init__(self, base: GraphSamplerBase):
        super().__init__(base.n_nodes, base.device, base.strict_n_nodes)
        self.base = base

    def forward(self) -> Data:
        graph = self.base.sample()
        return _to_simple(graph, self.device)


class DisjointUnion(GraphSamplerBase):
    def __init__(self, samplers: Sequence[GraphSamplerBase]):
        n_nodes = sum([g.n_nodes for g in samplers])
        device = samplers[0].device
        assert all([g.device == device for g in samplers])

        super().__init__(n_nodes, device)
        self.samplers = samplers

    def forward(self) -> Data:
        graphs = [sampler.sample() for sampler in self.samplers]
        edge_indices = []
        offset = 0
        for g in graphs:
            edge_indices.append(g.edge_index + offset)
            offset += g.num_nodes
        edge_index = torch.cat(edge_indices, dim=1)
        graph = _make_graph(edge_index, num_nodes=offset, device=self.device)
        graph = shuffle_graph(graph)
        return graph


class Merge(GraphSamplerBase):
    def __init__(self, samplers: Sequence[GraphSamplerBase]):
        n_nodes = samplers[0].n_nodes
        device = samplers[0].device
        assert all([g.device == device for g in samplers])
        assert all([g.n_nodes == n_nodes for g in samplers])

        super().__init__(n_nodes, device)
        self.samplers = samplers

    def forward(self) -> Data:
        graphs = [sampler.sample() for sampler in self.samplers]
        edge_index = torch.cat([g.edge_index for g in graphs], dim=1)
        edge_index = coalesce(edge_index, num_nodes=self.n_nodes)
        return _make_graph(edge_index, num_nodes=self.n_nodes, device=self.device)


# <<<


# >>> End-to-end samplers


class MultiLevelSbmWithPaSampler(GraphSamplerBase):
    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        min_pa_nodes_ratio: float = 0.0,
        max_pa_nodes_ratio: float = 0.3,
        max_pa_degree: int = 2,
        min_n_first_level_subgraphs: int = 10,
        max_n_first_level_subgraphs: int = 30,
        min_first_level_degree_ratio: float = 0.5,
        max_first_level_degree_ratio: float = 0.95,
        min_first_level_n_nodes: int = 16,
        device: torch.device | str,
    ):
        device = torch.device(device)

        super().__init__(n_nodes, device, strict_n_nodes=False)

        pa_nodes_ratio = random_float(min_pa_nodes_ratio, max_pa_nodes_ratio)
        n_nodes_before_pa = int(n_nodes * (1.0 - pa_nodes_ratio))

        pa_avg_degree = (1.0 + max_pa_degree) / 2
        avg_degree = (avg_degree - pa_nodes_ratio * pa_avg_degree) / (
            1.0 - pa_nodes_ratio
        )

        max_first_level_degree_ratio = min(
            max_first_level_degree_ratio, (min_first_level_n_nodes - 1) / avg_degree
        )
        first_level_degree_ratio = random_float(
            min_first_level_degree_ratio, max_first_level_degree_ratio
        )
        first_level_avg_degree = avg_degree * first_level_degree_ratio
        second_level_avg_degree = avg_degree * (1.0 - first_level_degree_ratio)

        # First-level sampler
        n_first_level_subgraphs = np.random.randint(
            min_n_first_level_subgraphs,
            max_n_first_level_subgraphs + 1,
        )

        first_level_subgraph_samplers = [
            DegreeCorrectedSBMSampler(
                n_nodes_subgraph,
                avg_degree=first_level_avg_degree,
                device=device,
            )
            for n_nodes_subgraph in random_partition(
                n=n_nodes_before_pa,
                n_parts=n_first_level_subgraphs,
                min_size=min_first_level_n_nodes,
            )
        ]

        first_level_sampler = DisjointUnion(first_level_subgraph_samplers)

        # Second-level sampler
        second_level_sampler = DegreeCorrectedSBMSampler(
            n_nodes_before_pa,
            avg_degree=second_level_avg_degree,
            device=device,
        )

        # Merge & apply PA
        sampler = Merge([first_level_sampler, second_level_sampler])
        sampler = ExtractLargestComponent(sampler)
        sampler = PreferentialAttachmentSampler(
            n_nodes,
            device,
            max_degree=max_pa_degree,
            base=sampler,
        )

        self.sampler = sampler

    def forward(self) -> Data:
        return self.sampler.sample()


class GraphSampler(GraphSamplerBase):
    _ML_SBM_PA_KEYS = (
        "min_pa_nodes_ratio",
        "max_pa_nodes_ratio",
        "max_pa_degree",
        "min_n_first_level_subgraphs",
        "max_n_first_level_subgraphs",
        "min_first_level_degree_ratio",
        "max_first_level_degree_ratio",
        "min_first_level_n_nodes",
    )

    def __init__(
        self,
        n_nodes: int,
        *,
        avg_degree: float,
        sampler_type: str,
        device: torch.device | str,
        sampler_kwargs: dict | None = None,
    ):
        device = torch.device(device)
        super().__init__(n_nodes, device, strict_n_nodes=False)

        extra = dict(sampler_kwargs or {})
        sampler_cls = {
            "multi-level-sbm-with-pa": MultiLevelSbmWithPaSampler,
            "sbm": DegreeCorrectedSBMSampler,
            "geometric": GeometricGraphSampler,
            "preferential-attachment": PreferentialAttachmentSampler,
            "erdos-renyi": ERSampler,
            "watts-strogatz": WattsStrogatzSampler,
        }[sampler_type]

        if sampler_type == "multi-level-sbm-with-pa":
            ml_kwargs = {
                k: extra[k]
                for k in self._ML_SBM_PA_KEYS
                if k in extra
            }
            sampler = sampler_cls(
                n_nodes,
                avg_degree=avg_degree,
                device=device,
                **ml_kwargs,
            )
        else:
            sampler = sampler_cls(n_nodes, avg_degree=avg_degree, device=device)  # type: ignore

        # Postprocessing & sanity checks
        sampler = ExtractLargestComponent(sampler)
        sampler = ToSimple(sampler)

        assert sampler.n_nodes == n_nodes
        self.sampler = sampler

    def forward(self) -> Data:
        return self.sampler.sample()


def sample_erdos_renyi_graph(
    n_nodes: int,
    avg_degree: float,
    device: torch.device | str,
) -> Data:
    """Erdős–Rényi-style graph with fixed ``n_nodes`` (no largest-component extraction).

    Same construction as ``GraphSampler(..., sampler_type="erdos-renyi")`` followed by
    :class:`ToSimple`, but without :class:`ExtractLargestComponent`, so node count
    matches ``n_nodes`` for alignment with node-aligned tensors.
    """
    device = torch.device(device)
    return ToSimple(ERSampler(n_nodes, avg_degree=avg_degree, device=device)).sample()


class BipartiteLatentFactorSampler:
    """Directed bipartite (user x item) recommendation graph from shared latents.

    Implements the "shared latent-factor model" (Option B): users and items get
    latent vectors ``z`` in a common space; the edge logit is the dot product
    ``z_u . z_i`` plus a per-item popularity bias (Zipf), and observed node
    features are a single shared linear projection of the latents plus noise
    (so user/item feature supports overlap, matching real AnyGraph bipartite
    datasets). Node ids are ``[users (0..U-1) | items (U..U+I-1)]`` and edges are
    directed ``user -> item``; only item nodes are ranked as candidates.

    ``sample()`` returns ``(edge_index, X, num_users, num_nodes)``.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        num_features: int,
        avg_degree: float,
        user_item_ratio: float,
        latent_dim_k: float,
        popularity_zipf_gamma: float,
        feature_informativeness: float,
        device: torch.device,
    ):
        self.n_nodes = max(4, int(n_nodes))
        self.num_features = max(1, int(num_features))
        self.avg_degree = max(1.0, float(avg_degree))
        self.user_item_ratio = max(1e-2, float(user_item_ratio))
        self.latent_dim_k = max(2, int(round(float(latent_dim_k))))
        self.popularity_zipf_gamma = max(1.01, float(popularity_zipf_gamma))
        self.feature_informativeness = max(1e-3, float(feature_informativeness))
        self.device = device

    def _split_user_item(self) -> tuple[int, int]:
        # ratio = U / I, U + I = n_nodes
        n_items = int(round(self.n_nodes / (1.0 + self.user_item_ratio)))
        n_items = min(self.n_nodes - 2, max(2, n_items))
        n_users = self.n_nodes - n_items
        return n_users, n_items

    def sample(self) -> tuple[Tensor, Tensor, int, int]:
        device = self.device
        U, I = self._split_user_item()
        N = U + I
        K = self.latent_dim_k

        z_u = torch.randn(U, K, device=device)
        z_i = torch.randn(I, K, device=device)

        pop = np.random.zipf(self.popularity_zipf_gamma, size=I).astype(np.float64)
        pop = np.clip(pop, 1.0, None)
        bias = torch.tensor(np.log(pop), dtype=torch.float32, device=device)
        bias = bias - bias.mean()

        logits = (z_u @ z_i.t()) / (K ** 0.5) + bias[None, :]
        probs = torch.sigmoid(logits)

        target_edges = 0.5 * N * self.avg_degree
        scale = target_edges / max(probs.sum().item(), 1e-8)
        probs = (probs * scale).clamp(max=1.0)
        adj = torch.rand_like(probs) < probs

        # Guarantee a connected-ish, non-empty graph: every user keeps at least
        # its highest-logit item edge.
        if adj.sum() == 0 or (~adj.any(dim=1)).any():
            best_item = logits.argmax(dim=1)
            adj[torch.arange(U, device=device), best_item] = True

        u_idx, i_idx = torch.nonzero(adj, as_tuple=True)
        src = u_idx
        dst = i_idx + U
        edge_index = torch.stack([src, dst], dim=0).to(device)

        # Shared latent -> feature projection (same W for users and items) + noise.
        F = self.num_features
        W = torch.randn(K, F, device=device) * (self.feature_informativeness / (K ** 0.5))
        z = torch.cat([z_u, z_i], dim=0)
        X = z @ W + torch.randn(N, F, device=device)

        return edge_index, X, U, N


# <<<


# >>> Data-conditioned graph samplers (features-first mode)


class DataConditionedSamplerBase(nn.Module, metaclass=abc.ABCMeta):
    """Base class for graph samplers that build a graph conditioned on node
    features ``X`` and labels ``y``."""

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        avg_degree: float,
        device: torch.device,
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.n_nodes = X.shape[0]
        self.avg_degree = avg_degree
        self.device = device

    @abc.abstractmethod
    def forward(self) -> Data:
        raise NotImplementedError()

    def sample(self) -> Data:
        graph = self.forward()
        assert _graph_device(graph) == self.device
        return graph


class LabelSBMSampler(DataConditionedSamplerBase):
    """Stochastic block model where communities are derived from node labels.

    Parameters
    ----------
    X : Tensor
        Node features (unused, present for interface consistency).
    y : Tensor
        Node labels of shape ``(n_nodes,)``.  Unique values define communities.
    avg_degree : float
        Target average degree.
    homophily_ratio : float
        In [0, 1].  1 = strongly homophilic (high intra, low inter),
        0 = strongly heterophilic (high inter, low intra).
    noise : float
        Fraction of nodes whose community assignment is randomly reassigned.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        *,
        avg_degree: float,
        homophily_ratio: float = 0.5,
        noise: float = 0.0,
        device: torch.device,
    ):
        super().__init__(X, y, avg_degree, device)
        self.homophily_ratio = homophily_ratio
        self.noise = noise

    def forward(self) -> Data:
        y = self.y.clone().to(self.device)
        n = self.n_nodes

        # --- assign communities from labels, with optional noise ---
        unique_labels = y.unique()
        n_communities = len(unique_labels)
        label_to_community = {lbl.item(): i for i, lbl in enumerate(unique_labels)}
        communities = torch.tensor(
            [label_to_community[lbl.item()] for lbl in y],
            device=self.device,
        )
        if self.noise > 0 and n_communities > 1:
            flip_mask = torch.rand(n, device=self.device) < self.noise
            random_communities = torch.randint(0, n_communities, (n,), device=self.device)
            communities = torch.where(flip_mask, random_communities, communities)

        # --- compute intra / inter community edge probabilities ---
        group_sizes = torch.bincount(communities, minlength=n_communities).float()

        # Expected number of intra-community node pairs (upper-triangular only)
        intra_pairs = 0.5 * (group_sizes * (group_sizes - 1)).sum()
        # Expected number of inter-community node pairs
        inter_pairs = 0.5 * n * (n - 1) - intra_pairs

        target_edges = 0.5 * n * self.avg_degree
        h = self.homophily_ratio

        # Distribute edges between intra and inter, guarded against 0-pair cases
        if intra_pairs < 1:
            p_intra, p_inter = 0.0, (target_edges / max(inter_pairs, 1.0)).item()
        elif inter_pairs < 1:
            p_intra, p_inter = (target_edges / max(intra_pairs, 1.0)).item(), 0.0
        else:
            intra_edge_share = h * target_edges
            inter_edge_share = (1 - h) * target_edges
            p_intra = (intra_edge_share / intra_pairs).item()
            p_inter = (inter_edge_share / inter_pairs).item()

        p_intra = min(p_intra, 1.0)
        p_inter = min(p_inter, 1.0)

        # --- build block probability matrix & sample ---
        same_community = communities[:, None] == communities[None, :]
        probs = torch.where(
            same_community,
            torch.tensor(p_intra, device=self.device),
            torch.tensor(p_inter, device=self.device),
        ).triu(diagonal=1)

        adj = (torch.rand_like(probs) < probs).to(torch.int32)
        graph = graph_from_adj(adj, device=self.device)
        return graph


def sample_featureless_label_sbm(
    y: Tensor,
    *,
    p_same: float,
    p_diff: float,
    device: torch.device,
) -> Data:
    """Undirected planted-partition SBM with explicit same/diff edge probabilities.

    Same-label pairs are connected independently with probability ``p_same``,
    other pairs with ``p_diff``. The full node set is kept (no LCC shrink).
    If the first draw has no edges and at least one probability is positive,
    edges are resampled once.
    """
    y = y.to(device)
    n = int(y.shape[0])
    if n < 2:
        return _make_graph(
            torch.empty(2, 0, dtype=torch.long, device=device),
            num_nodes=n,
            device=device,
        )

    p_same_f = float(p_same)
    p_diff_f = float(p_diff)
    same = y[:, None] == y[None, :]
    probs = torch.where(
        same,
        torch.tensor(p_same_f, device=device),
        torch.tensor(p_diff_f, device=device),
    ).triu(diagonal=1)

    def _draw() -> Data:
        adj = (torch.rand_like(probs) < probs).to(torch.int32)
        return graph_from_adj(adj, device=device)

    graph = _draw()
    if graph.edge_index.numel() == 0 and (p_same_f > 0.0 or p_diff_f > 0.0):
        graph = _draw()
    return graph


def _perturb_kernel_adjacency_reassign_edges(
    adj: Tensor,
    edge_perturb_prob: float,
    *,
    device: torch.device,
) -> Tensor:
    """For each existing upper-triangular edge, with probability ``edge_perturb_prob``
    remove that edge and insert a new edge between two uniformly random distinct
    nodes.  Helps connectivity when thresholding yields a fragmented graph."""
    if edge_perturb_prob <= 0:
        return adj
    n = adj.shape[0]
    if n < 2:
        return adj
    triu_idx = torch.triu_indices(n, n, offset=1, device=device)
    edge_mask = adj[triu_idx[0], triu_idx[1]] > 0
    if not edge_mask.any():
        return adj
    ei = triu_idx[0][edge_mask]
    ej = triu_idx[1][edge_mask]
    m = ei.shape[0]
    perturb = torch.rand(m, device=device) < edge_perturb_prob
    if not perturb.any():
        return adj
    adj = adj.clone()
    pi, pj = ei[perturb], ej[perturb]
    adj[pi, pj] = 0
    n_new = int(perturb.sum().item())
    u = torch.randint(0, n, (n_new,), device=device)
    v = torch.randint(0, n, (n_new,), device=device)
    same = u == v
    while same.any():
        v[same] = torch.randint(
            0, n, (int(same.sum().item()),), device=device
        )
        same = u == v
    ri = torch.minimum(u, v)
    rj = torch.maximum(u, v)
    adj[ri, rj] = 1
    return adj


class KernelGraphSampler(DataConditionedSamplerBase):
    """Builds a graph by computing pairwise kernel values from node features.

    Parameters
    ----------
    X : Tensor
        Node features of shape ``(n_nodes, d)``.
    y : Tensor
        Node labels (unused, present for interface consistency).
    avg_degree : float
        Target average degree.
    kernel : str
        One of ``"rbf"``, ``"cosine"``, ``"inner-product"``.
    homophily_ratio : float
        In [0, 1].  1 = homophilic (similar nodes connected),
        0 = heterophilic (dissimilar nodes connected).
    edge_method : str
        ``"threshold"`` (deterministic) or ``"bernoulli"`` (stochastic).
    edge_perturb_prob : float
        After building the adjacency, each existing edge is removed and replaced
        by a uniform random edge between two distinct nodes with this probability
        (independently per edge).  Improves connectivity when ``threshold`` gives
        many small components.  ``0`` disables.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        *,
        avg_degree: float,
        kernel: str = "rbf",
        homophily_ratio: float = 0.5,
        edge_method: str = "threshold",
        edge_perturb_prob: float = 0.0,
        device: torch.device,
    ):
        super().__init__(X, y, avg_degree, device)
        self.kernel = kernel
        self.homophily_ratio = homophily_ratio
        self.edge_method = edge_method
        self.edge_perturb_prob = edge_perturb_prob

    def _compute_kernel(self, X: Tensor) -> Tensor:
        if self.kernel == "rbf":
            dists = torch.cdist(X[None], X[None])[0]
            median_dist = dists[dists > 0].median()
            bandwidth = max(median_dist.item(), 1e-6) * 3
            K = torch.exp(-dists**2 / (2 * bandwidth**2))
        elif self.kernel == "cosine":
            norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
            X_normed = X / norms
            K = X_normed @ X_normed.T
            K = (K + 1) / 2  # shift to [0, 1]
        elif self.kernel == "inner-product":
            K = X @ X.T
            K = (K - K.min()) / (K.max() - K.min() + 1e-8)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        return K

    def forward(self) -> Data:
        X = self.X.float().to(self.device)
        n = self.n_nodes

        K = self._compute_kernel(X)

        # Apply homophily/heterophily: blend between K (homophilic) and 1-K
        K = self.homophily_ratio * K + (1 - self.homophily_ratio) * (1 - K)

        # Zero out diagonal (no self-loops) and keep upper triangle
        K = K.triu(diagonal=1)

        if self.edge_method == "threshold":
            if self.avg_degree < 4:
                self.avg_degree = 4
            target_edges = 0.5 * n * self.avg_degree
            flat = K[K > 0]
            subsample_size = 100_000
            if flat.numel() == 0:
                adj = torch.zeros((n, n), dtype=torch.int32, device=self.device)
            else:
                n_flat = flat.numel()
                q = 1.0 - target_edges / n_flat
                q = max(0.0, min(q, 1.0))
                if n_flat > subsample_size:
                    perm = torch.randperm(n_flat, device=flat.device)
                    flat = flat[perm][:subsample_size]
                threshold = torch.quantile(flat.float(), q).item()
                adj = (K >= threshold).to(torch.int32)
        elif self.edge_method == "bernoulli":
            target_edges = 0.5 * n * self.avg_degree
            total_pairs = 0.5 * n * (n - 1)
            scale = target_edges / max(K.sum().item(), 1e-8)
            probs = (K * scale).clamp(max=1.0)
            adj = (torch.rand_like(probs) < probs).to(torch.int32)
        else:
            raise ValueError(f"Unknown edge_method: {self.edge_method}")

        adj = _perturb_kernel_adjacency_reassign_edges(
            adj,
            self.edge_perturb_prob,
            device=self.device,
        )

        graph = graph_from_adj(adj, device=self.device)
        return graph


class LabelOrderedWattsStrogatzSampler(DataConditionedSamplerBase):
    """Watts-Strogatz small-world graph on a ring whose node order follows labels.

    Nodes are placed on the circle in **sorted label order** (ties broken by
    original index).  Optionally, a fraction of nodes are chosen at random and
    their **ordering label** is replaced by a uniformly sampled **different**
    class from ``y.unique()`` before sorting; true ``y`` is unchanged for
    alignment with ``X``.  A standard WS process (``k``-nearest ring neighbours,
    rewiring probability ``beta``) runs in that ring order; edges are mapped
    back to original node ids ``0..n-1``.

    ``X`` is unused; only ``y`` defines the circle layout.
    """

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        *,
        avg_degree: float,
        beta_min: float = 0.01,
        beta_max: float = 0.3,
        label_order_flip_fraction: float = 0.0,
        device: torch.device,
    ):
        super().__init__(X, y, avg_degree, device)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.label_order_flip_fraction = label_order_flip_fraction

    def forward(self) -> Data:
        import networkx as nx

        n = self.n_nodes
        device = self.device
        y_ord = self.y.float().view(-1).to(device)
        unique = torch.unique(y_ord, sorted=True)
        frac = float(
            min(1.0, max(0.0, self.label_order_flip_fraction))
        )
        if frac > 0 and unique.numel() > 1 and n > 0:
            k = int(round(frac * n))
            k = min(n, max(0, k))
            if k > 0:
                flip_idx = torch.randperm(n, device=device)[:k]
                for idx in flip_idx:
                    yi = y_ord[idx]
                    others = unique[unique != yi]
                    if others.numel() > 0:
                        pick = torch.randint(0, others.numel(), (1,), device=device)
                        y_ord[idx] = others[pick]

        order = torch.argsort(
            y_ord * float(n + 1)
            + torch.arange(n, device=device, dtype=y_ord.dtype)
        )

        k = max(2, int(self.avg_degree))
        if k % 2 != 0:
            k += 1
        beta = random_float(self.beta_min, self.beta_max)

        G = nx.watts_strogatz_graph(n, k, beta)
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            return _make_graph(edge_index, num_nodes=n, device=self.device)

        src_ring = torch.tensor([e[0] for e in edges], device=self.device)
        dst_ring = torch.tensor([e[1] for e in edges], device=self.device)
        src = order[src_ring.long()]
        dst = order[dst_ring.long()]

        return create_graph_from_edgelist(
            src, dst, device=self.device, n_nodes=n
        )


class DataConditionedGraphSampler(nn.Module):
    """Factory that dispatches to ``LabelSBMSampler`` or ``KernelGraphSampler``
    and applies standard postprocessing (LCC extraction + simplification).

    Parameters
    ----------
    X : Tensor
        Node features of shape ``(n_nodes, d)``.
    y : Tensor
        Node labels of shape ``(n_nodes,)``.
    avg_degree : float
        Target average degree.
    sampler_type : str
        One of ``"label-sbm"``, ``"label-watts-strogatz"``, ``"kernel-rbf"``,
        ``"kernel-cosine"``, ``"kernel-inner-product"``.
    homophily_ratio : float
        Controls homophily (1) vs heterophily (0).
    sbm_noise : float
        Community-flip noise for ``LabelSBMSampler``.
    edge_method : str
        ``"threshold"`` or ``"bernoulli"`` for kernel samplers.
    kernel_edge_perturb_prob : float
        Per-edge random reassignment probability for kernel samplers (see
        :class:`KernelGraphSampler`).  Ignored for ``label-sbm`` and
        ``label-watts-strogatz``.
    ws_beta_min, ws_beta_max : float
        Rewiring probability range for ``label-watts-strogatz`` (same role as
        :class:`WattsStrogatzSampler`).
    ws_label_order_flip_fraction : float
        Fraction of nodes (``round(frac * n)``) whose **ordering** label is
        randomly reassigned to another class before sorting; ``0`` disables.
        Only affects ``label-watts-strogatz``.
    device : torch.device or str
        Target device.
    """

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        *,
        avg_degree: float,
        sampler_type: str,
        homophily_ratio: float = 0.5,
        sbm_noise: float = 0.0,
        edge_method: str = "threshold",
        kernel_edge_perturb_prob: float = 0.0,
        ws_beta_min: float = 0.01,
        ws_beta_max: float = 0.3,
        ws_label_order_flip_fraction: float = 0.0,
        device: torch.device | str,
    ):
        super().__init__()
        device = torch.device(device)

        if sampler_type == "label-sbm":
            self.base_sampler = LabelSBMSampler(
                X, y,
                avg_degree=avg_degree,
                homophily_ratio=homophily_ratio,
                noise=sbm_noise,
                device=device,
            )
        elif sampler_type == "label-watts-strogatz":
            self.base_sampler = LabelOrderedWattsStrogatzSampler(
                X, y,
                avg_degree=avg_degree,
                beta_min=ws_beta_min,
                beta_max=ws_beta_max,
                label_order_flip_fraction=ws_label_order_flip_fraction,
                device=device,
            )
        elif sampler_type.startswith("kernel-"):
            kernel = sampler_type.split("kernel-", 1)[1]
            self.base_sampler = KernelGraphSampler(
                X, y,
                avg_degree=avg_degree,
                kernel=kernel,
                homophily_ratio=homophily_ratio,
                edge_method=edge_method,
                edge_perturb_prob=kernel_edge_perturb_prob,
                device=device,
            )
        else:
            raise ValueError(f"Unknown data-conditioned sampler type: {sampler_type}")

        self.device = device

    def sample(self) -> tuple[Data, Tensor]:
        graph = self.base_sampler.sample()
        graph, component_nodes = extract_largest_component(graph)
        graph = _to_simple(graph, self.device)
        return graph, component_nodes


# <<<
