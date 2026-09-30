"""Transform to remove self-loops from graph edges."""

import torch
from torch_geometric.data import Data


class RemoveSelfLoops:
    """Remove self-loops (edges where source == target) from the graph.

    This transform filters out any edges where the source and target node
    are the same, which is typically desired for message passing networks.

    Also remaps ``inverse_edge_map`` / edge-level ``visible_mask`` when present,
    so KG inverse-edge pairing stays consistent after filtering.
    """

    def __call__(self, data: Data) -> Data:
        """Apply the transform to remove self-loops.

        Args:
            data: PyG Data object with edge_index and optionally edge_type.

        Returns:
            Data object with self-loops removed.
        """
        edge_index = data.edge_index

        # Create mask for non-self-loop edges
        mask = edge_index[0] != edge_index[1]
        if bool(mask.all()):
            return data

        num_edges = int(mask.numel())
        old_to_new = torch.full((num_edges,), -1, dtype=torch.long, device=mask.device)
        old_to_new[mask] = torch.arange(int(mask.sum().item()), device=mask.device)

        # Filter edge_index
        data.edge_index = edge_index[:, mask]

        # Filter edge_type if it exists
        if hasattr(data, "edge_type") and data.edge_type is not None:
            data.edge_type = data.edge_type[mask]

        # Filter edge_attr if it exists
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]

        # Filter edge-level split masks if they exist (skip node-level masks)
        for mask_name in ("train_mask", "val_mask", "test_mask", "visible_mask"):
            if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
                attr = getattr(data, mask_name)
                if attr.size(0) == num_edges:
                    setattr(data, mask_name, attr[mask])

        inv_map = getattr(data, "inverse_edge_map", None)
        if inv_map is not None and inv_map.numel() == num_edges:
            kept_old = mask.nonzero(as_tuple=True)[0]
            new_inv = old_to_new[inv_map[kept_old]]
            if (new_inv < 0).any():
                raise RuntimeError(
                    "RemoveSelfLoops: inverse of a kept edge was removed; "
                    "inverse_edge_map cannot be remapped."
                )
            data.inverse_edge_map = new_inv

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
