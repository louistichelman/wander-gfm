"""Transform to add inverse edges with distinct edge types."""

import torch
from torch_geometric.data import Data


class AddInverseEdges:
    """Add inverse edges for each edge with a new edge type.
    
    For each edge (u, v) with type r, adds an inverse edge (v, u) with type r + num_relations.
    If no edge types exist, assigns type 0 to original edges and type 1 to inverse edges.
    
    This transform doubles the number of edges and relation types.
    
    Also creates an inverse_edge_map lookup table where inverse_edge_map[i] gives
    the index of the inverse edge for edge i.
    """
    
    def __call__(self, data: Data) -> Data:
        """Apply the transform to add inverse edges.
        
        Args:
            data: PyG Data object with edge_index and optionally edge_type.
            
        Returns:
            Data object with inverse edges added, edge_type assigned,
            and inverse_edge_map lookup table.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        device = edge_index.device
        
        # Handle edge types
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            edge_type = data.edge_type
            num_relations = data.num_relations if hasattr(data, 'num_relations') else edge_type.max().item() + 1
        else:
            # No existing edge types - assign type 0 to all original edges
            edge_type = torch.zeros(num_edges, dtype=torch.long, device=device)
            num_relations = 1
        
        # Create inverse edges (swap source and target)
        inverse_edge_index = edge_index.flip(0)
        
        # Inverse edge types are offset by num_relations
        inverse_edge_type = edge_type + num_relations
        
        # Create inverse edge lookup table
        # inverse_edge_map[i] = index of inverse edge for edge i
        # Original edges (0 to num_edges-1) map to inverse edges (num_edges to 2*num_edges-1)
        # Inverse edges (num_edges to 2*num_edges-1) map back to original edges (0 to num_edges-1)
        inverse_edge_map = torch.cat([
            torch.arange(num_edges, 2 * num_edges, device=device),  # originals -> inverses
            torch.arange(num_edges, device=device),                  # inverses -> originals
        ])
        data.inverse_edge_map = inverse_edge_map
        
        # Concatenate original and inverse edges
        data.edge_index = torch.cat([edge_index, inverse_edge_index], dim=1)
        data.edge_type = torch.cat([edge_type, inverse_edge_type], dim=0)
        data.num_relations = num_relations * 2
        
        # Handle edge_attr if it exists (duplicate for inverse edges)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
        
        # # Handle augmentation_edge_mask if it exists (duplicate for inverse edges)
        # if hasattr(data, 'augmentation_edge_mask') and data.augmentation_edge_mask is not None:
        #     data.augmentation_edge_mask = torch.cat([
        #         data.augmentation_edge_mask,
        #         data.augmentation_edge_mask
        #     ], dim=0)
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
