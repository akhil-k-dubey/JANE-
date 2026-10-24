"""Relational Manifold: 256D Geometric Concept Workspace with 3D Presentation Head."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PoincareGeometry:
    """Hyperbolic utilities to model hierarchical tree-like concepts cleanly."""

    def __init__(self, curvature: float = 1.0, eps: float = 1e-5):
        self.curvature = curvature
        self.eps = eps

    @property
    def radius(self) -> float:
        return 1.0 / math.sqrt(self.curvature)

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Projects points into the interior of the Poincare ball."""
        norm = points.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        max_norm = self.radius * (1.0 - self.eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return points * scale

    def pairwise_distance(self, points: torch.Tensor) -> torch.Tensor:
        points = self.project(points)
        norm_sq = (points * points).sum(dim=-1).clamp_max((self.radius * (1.0 - self.eps)) ** 2)
        diff = points.unsqueeze(2) - points.unsqueeze(1)
        diff_sq = (diff * diff).sum(dim=-1)
        denom = (1.0 - self.curvature * norm_sq).unsqueeze(2) * (1.0 - self.curvature * norm_sq).unsqueeze(1)
        arg = 1.0 + 2.0 * self.curvature * diff_sq / denom.clamp_min(self.eps)
        return torch.acosh(arg.clamp_min(1.0 + self.eps)) / math.sqrt(self.curvature)


class RelationalManifold(nn.Module):
    """
    The 256D Relational Manifold (The "Dots" and Connections).
    An expressive geometric graph workspace where concepts, entities, and intermediate
    variables interact via learned relational message passing.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_nodes: int = 32,
        num_layers: int = 3,
        geometry: str = "euclidean",
        presentation_dim: int = 3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.geometry_type = geometry
        self.ball = PoincareGeometry() if geometry == "poincare" else None

        # Base concept dot priors in the manifold
        self.node_priors = nn.Parameter(torch.randn(num_nodes, latent_dim) * 0.02)
        self.context_injector = nn.Linear(latent_dim, num_nodes * latent_dim)

        # Multi-layer relational message passing
        self.msg_transforms = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_layers)
        ])
        self.node_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim * 2),
                nn.GELU(),
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim)
            ) for _ in range(num_layers)
        ])

        # Presentation Head: Projects 256D thoughts down to 3D for Stark-style holographic visualization
        self.presentation_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, presentation_dim)
        )

    def forward(
        self,
        context_latent: torch.Tensor,
        active_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Form dynamic connections between dots based on context.
        Supports dynamic adaptive node sizing: if active_k is specified,
        only the relevant active subset of concept nodes is processed.
        Args:
            context_latent: [batch, latent_dim]
            active_k: optional int to dynamically scale active node count (e.g. 8 for simple, 32 for full)
        Returns:
            nodes: [batch, k, latent_dim]
            adjacency: [batch, k, k] dynamic relational graph
        """
        batch = context_latent.size(0)
        offsets = self.context_injector(context_latent).view(batch, self.num_nodes, self.latent_dim)
        nodes = self.node_priors.unsqueeze(0) + offsets

        if active_k is not None and 2 <= active_k <= self.num_nodes:
            nodes = nodes[:, :active_k, :]

        if self.ball:
            nodes = self.ball.project(nodes)

        # Compute dynamic relational edges (soft adjacency)
        if self.ball:
            dist = self.ball.pairwise_distance(nodes)
            adjacency = F.softmax(-dist, dim=-1)
        else:
            q = nodes
            k = nodes
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.latent_dim)
            adjacency = F.softmax(scores, dim=-1)

        # Message passing rounds with residual connections
        for msg_fn, update_fn in zip(self.msg_transforms, self.node_updates):
            messages = torch.matmul(adjacency, msg_fn(nodes))
            delta = update_fn(torch.cat([nodes, messages], dim=-1))
            nodes = nodes + delta
            if self.ball:
                nodes = self.ball.project(nodes)

        return nodes, adjacency

    def readout(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compresses graph dots into a single unified 256D concept representation."""
        return nodes.mean(dim=1)

    def project_to_3d(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Projects 256D manifold nodes into 3D coordinates for visual presentation.
        Returns:
            coords_3d: [batch, num_nodes, 3]
        """
        return self.presentation_head(nodes)
