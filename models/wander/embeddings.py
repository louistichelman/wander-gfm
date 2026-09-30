import math
import torch
from torch import nn, Tensor


class LabelEmbedding(nn.Module):
    """Learnable embedding table for categorical node labels.

    Stores ``max_num_classes + 1`` vectors: one per class plus a mask token
    (index ``max_num_classes``).  Initialized with mutually orthogonal vectors
    via ``torch.nn.init.orthogonal_`` (requires ``max_num_classes + 1 <= dim``).
    """

    def __init__(self, max_num_classes: int, dim: int):
        super().__init__()

        assert max_num_classes + 1 <= dim, (
            f"max_num_classes + 1 ({max_num_classes + 1}) must be <= dim ({dim}) "
            "for orthogonal initialization"
        )
        self.max_num_classes = max_num_classes
        self.embedding = nn.Embedding(max_num_classes + 1, dim) # max_num_classes + 1 is the mask token
        nn.init.orthogonal_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight.mul_(math.sqrt(dim))

    def forward(self, labels: Tensor) -> Tensor:
        """
        Args:
            labels: ``[...]`` integer label indices (0..max_num_classes).
                    Index ``max_num_classes`` is the mask token.
        Returns:
            ``[..., dim]``
        """

        return self.embedding(labels)


class FeatureEmbedding(nn.Module):
    """Embeds per-feature scalar values into D-dimensional vectors.

    By default applies a single ``Linear`` layer to each feature slot. When
    ``use_mlp=True`` a 2-layer MLP (``Linear -> SiLU -> Linear``) is used
    instead.

    When ``feature_groups > 1`` features are embedded in groups of that size:
    the last feature dimension is padded with zeros to a multiple of
    ``feature_groups`` and reshaped into ``[..., num_groups, feature_groups]``
    before embedding, where ``num_groups = ceil(num_features / feature_groups)``.
    """

    def __init__(self, dim: int, feature_groups: int = 1, use_mlp: bool = False):
        super().__init__()
        assert feature_groups >= 1
        self.feature_groups = feature_groups
        self.use_mlp = use_mlp
        in_dim = feature_groups
        if use_mlp:
            # Named ``mlp`` (not ``embed``) so checkpoints from before the embed refactor load.
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        else:
            self.embed = nn.Linear(in_dim, dim)

    def forward(self, values: Tensor) -> Tensor:
        """
        Args:
            values: ``[..., num_features]`` raw scalar feature values.
        Returns:
            ``[..., num_groups, dim]`` where
            ``num_groups = ceil(num_features / feature_groups)``.
        """
        body = self.mlp if self.use_mlp else self.embed
        if self.feature_groups == 1:
            return body(values.unsqueeze(-1))
        num_features = values.shape[-1]
        pad = (-num_features) % self.feature_groups
        if pad:
            pad_shape = list(values.shape)
            pad_shape[-1] = pad
            values = torch.cat([values, values.new_zeros(pad_shape)], dim=-1)
        values = values.view(*values.shape[:-1], -1, self.feature_groups)
        return body(values)
