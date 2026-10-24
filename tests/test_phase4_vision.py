"""Unit & Integration Tests for Phase 4: Multimodal Embodiment (Vision Encoder)."""

import pytest
import torch

from src.perception.vision_encoder import VisionPatchTransformer
from environments.spatial_graph.environment import SpatialEnvironmentGraph, ContinuousObstacle
from environments.spatial_graph.visual_renderer import FacilityVisualRenderer
from environments.spatial_graph.continuous_kinematics import ContinuousKinematicsEngine
from src.translator.pretrained_translator import PretrainedLinguisticEar
from src.reasoning_engine.manifold import RelationalManifold


def test_vision_patch_transformer_dimensions():
    encoder = VisionPatchTransformer(
        img_size=128,
        patch_size=16,
        in_channels=3,
        vision_dim=192,
        num_layers=2,
        num_heads=4,
        latent_dim=256
    )

    dummy_img = torch.randn(2, 3, 128, 128)

    # 1. Forward pass produces exact 256D latent concept vectors
    z_visual = encoder(dummy_img)
    assert z_visual.shape == (2, 256)
    assert torch.isfinite(z_visual).all()

    # 2. Extract patch tokens: 128/16 = 8 -> 8x8 = 64 spatial patches
    cls_token, spatial_patches = encoder.extract_patch_tokens(dummy_img)
    assert cls_token.shape == (2, 192)
    assert spatial_patches.shape == (2, 64, 192)


def test_facility_visual_renderer():
    env = SpatialEnvironmentGraph.create_facility_env()
    renderer = FacilityVisualRenderer(img_size=128)

    # 1. Render clean environment
    img = renderer.render_environment(env)
    assert img.shape == (3, 128, 128)
    assert (img >= 0.0).all() and (img <= 1.0).all()

    # 2. Render with injected hallucinated door
    img_hallucinated = renderer.render_environment(env, inject_fake_door_through_obstacle=True)
    assert img_hallucinated.shape == (3, 128, 128)
    # The hallucinated green door modifies image pixels
    assert not torch.equal(img, img_hallucinated)


def test_neuro_symbolic_vision_hallucination_firewall():
    """
    Core Safety Property:
    Even if computer vision hallucinates a door where there is a solid wall,
    the continuous kinematics solver mathematically blocks the trajectory.
    """
    env = SpatialEnvironmentGraph.create_facility_env()
    kinematics = ContinuousKinematicsEngine()

    # Vision model hallucinates that it can walk straight through the central_pillar at (2.75, 3.0)
    fake_shortcut_trajectory = [(0.0, 0.0), (2.75, 3.0), (5.0, 5.0)]

    # Symbolic proof check
    proof = kinematics.verify_trajectory(fake_shortcut_trajectory, env.obstacles)

    # Must be mathematically rejected!
    assert proof.is_valid is False
    assert proof.prover_name == "Continuous_Kinematics_Engine"
    assert "collides with obstacle" in proof.proof_log

    # Real obstacle detour is verified and approved
    safe_trajectory = kinematics.plan_continuous_trajectory((0.0, 0.0), (5.0, 5.0), env.obstacles)
    safe_proof = kinematics.verify_trajectory(safe_trajectory, env.obstacles)
    assert safe_proof.is_valid is True
