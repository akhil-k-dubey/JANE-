"""Vision Transformer (ViT) & Patch Encoder for Multimodal Embodiment.

The Vision Encoder does NOT reason. Its only job is to:
1. Extract visual spatial patch tokens from RGB images / camera feeds.
2. Project visual representations into the EXACT 256D manifold latent space
   used by the Pretrained Ear and the Knowledge Graph.
3. Align visual tokens with conceptual knowledge dots in the 3D manifold.
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionPatchTransformer(nn.Module):
    """
    Lightweight Vision Transformer (ViT) patch encoder.
    Maps an RGB image [B, 3, H, W] directly to a 256D continuous latent vector,
    and extracts spatially localized patch tokens for object identification.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        vision_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        latent_dim: int = 256,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim

        # 1. Patch Embedding via 2D Convolution
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=vision_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 2. Learnable Class Token and 1D Position Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, vision_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, vision_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            dim_feedforward=int(vision_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(vision_dim)

        # 4. Multimodal Bottleneck Head (Vision -> 256D Manifold)
        self.manifold_projection = nn.Sequential(
            nn.Linear(vision_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 5. Local Patch Projection Head (for localized object queries)
        self.patch_projection = nn.Linear(vision_dim, latent_dim)

    def extract_patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts visual tokens.
        Args:
            x: RGB image tensor [B, 3, H, W]
        Returns:
            global_token: [B, vision_dim] (CLS token)
            patch_tokens: [B, num_patches, vision_dim] (Spatial tokens)
        """
        batch = x.size(0)
        # Patch projection: [B, vision_dim, H/P, W/P] -> [B, num_patches, vision_dim]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches + 1, vision_dim]

        # Add positional embedding
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Pass through Transformer encoder blocks
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        global_token = encoded[:, 0]  # CLS
        spatial_patches = encoded[:, 1:]  # Patches
        return global_token, spatial_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image directly into a 256D latent concept vector.
        Args:
            x: RGB image tensor [B, 3, H, W]
        Returns:
            z_visual: [B, 256] continuous vector aligned with the Manifold
        """
        global_token, _ = self.extract_patch_tokens(x)
        z_visual = self.manifold_projection(global_token)
        return z_visual

    def detect_objects_in_patches(
        self,
        x: torch.Tensor,
        text_concept_latents: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Zero-shot visual grounding:
        Compares each spatial patch token against 256D text concept latents
        from the Knowledge Graph (e.g. 'key', 'hazard', 'door', 'reactor').
        Returns list of detected objects with continuous image patch locations.
        """
        _, spatial_patches = self.extract_patch_tokens(x)  # [B, num_patches, vision_dim]
        patch_256 = self.patch_projection(spatial_patches)  # [B, num_patches, 256]
        norm_patches = F.normalize(patch_256, p=2, dim=-1)

        grid_side = int(math.sqrt(self.num_patches))
        detections = []

        for concept_name, concept_vec in text_concept_latents.items():
            norm_concept = F.normalize(concept_vec.view(1, 1, 256).to(norm_patches.device), p=2, dim=-1)
            # Cosine similarity across all patches: [B, num_patches]
            sim = (norm_patches * norm_concept).sum(dim=-1)[0]
            max_sim, best_idx = torch.max(sim, dim=0)

            # Convert 1D patch index to 2D normalized image coordinate
            patch_idx = best_idx.item()
            row = patch_idx // grid_side
            col = patch_idx % grid_side
            norm_x = (col + 0.5) / grid_side
            norm_y = (row + 0.5) / grid_side

            detections.append({
                "concept": concept_name,
                "confidence": max_sim.item(),
                "patch_coord": (norm_x, norm_y),
                "patch_index": patch_idx
            })

        return detections
