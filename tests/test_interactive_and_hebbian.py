"""Unit Tests for Interactive Chat, Hebbian Manifold, and Mode Router."""

import pytest
import torch
import os
import shutil

from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.core.mode_router import ModeRouter, OperatingMode
from src.core.interactive_chat import JaneTerminalChat
from src.symbolic_verifier.contracts import MathHypothesis, PhysicsHypothesis, Dimension


@pytest.fixture
def tmp_manifold_path(tmp_path):
    return str(tmp_path / "test_manifold.pt")


class TestHebbianConceptManifold:

    def test_manifold_initialization(self, tmp_manifold_path):
        m = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        assert m.latent_dim == 256
        assert m.num_concepts > 20
        assert "energy" in m.concept_names
        assert "entropy" in m.concept_names
        assert m.concept_nodes.shape == (m.num_concepts, 256)
        assert m.adjacency.shape == (m.num_concepts, m.num_concepts)

    def test_process_acoustic_flux_hebbian_wiring(self, tmp_manifold_path):
        m = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        initial_adj = m.adjacency.clone()

        # Create a mock 256D acoustic vector close to 'energy' and 'entropy'
        idx_energy = m.name_to_idx["energy"]
        idx_entropy = m.name_to_idx["entropy"]

        # Blend energy and entropy nodes
        z_sample = (m.concept_nodes[idx_energy] + m.concept_nodes[idx_entropy]) / 2.0
        z_sample = z_sample.unsqueeze(0)

        # Process flux
        telemetry = m.process_acoustic_flux(
            z_audio=z_sample,
            audio_rms=0.045,
            top_k=4,
            firing_threshold=0.10
        )

        assert telemetry["utterance_count"] == 1
        assert telemetry["audio_rms"] == 0.045
        assert len(telemetry["fired_concepts"]) >= 2

        # Check that the edge weight between energy and entropy increased (Hebbian rule)
        w_before = initial_adj[idx_energy, idx_entropy].item()
        w_after = m.adjacency[idx_energy, idx_entropy].item()
        assert w_after >= w_before

        # Check tone classification
        assert telemetry["tone_label"] is not None

    def test_tone_modality_classification(self, tmp_manifold_path):
        m = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        z_rand = torch.randn(1, 256)

        # Very low RMS -> Whispering / Quiet Reflection
        tel_whisper = m.process_acoustic_flux(z_rand, audio_rms=0.015)
        assert "Whispering" in tel_whisper["tone_label"]

        # High RMS -> Emphatic / Teaching Loudly
        tel_loud = m.process_acoustic_flux(z_rand, audio_rms=0.095)
        assert "Teaching" in tel_loud["tone_label"] or "Emphatic" in tel_loud["tone_label"]

    def test_save_and_load_manifold(self, tmp_manifold_path):
        m1 = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        z_sample = torch.randn(1, 256)
        m1.process_acoustic_flux(z_sample, audio_rms=0.040)
        m1.save_manifold()

        assert os.path.exists(tmp_manifold_path)

        # Load in fresh instance
        m2 = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        assert m2.total_utterances_observed == 1
        assert torch.allclose(m1.concept_nodes, m2.concept_nodes)
        assert torch.allclose(m1.adjacency, m2.adjacency)

    def test_find_nearest_concepts_and_exclusion(self, tmp_manifold_path):
        m = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        v_energy = m.get_vector("energy")

        # 1. Exact match retrieval
        nearest = m.find_nearest_concepts(v_energy, top_k=3)
        assert len(nearest) == 3
        assert nearest[0][0] == "energy"
        assert pytest.approx(nearest[0][1], abs=1e-4) == 1.0

        # 2. Exclusion test
        nearest_ex = m.find_nearest_concepts(v_energy, top_k=3, exclude=["energy"])
        assert len(nearest_ex) == 3
        assert all(name != "energy" for name, _ in nearest_ex)

    def test_find_nearest_concepts_for_text(self, tmp_manifold_path):
        m = HebbianConceptManifold(latent_dim=256, save_path=tmp_manifold_path)
        nearest = m.find_nearest_concepts_for_text("thermodynamics entropy equilibrium", top_k=3)
        assert len(nearest) == 3
        names = [n[0] for n in nearest]
        # Should correlate with thermodynamic concepts
        assert any(c in names for c in ("thermodynamics", "entropy", "equilibrium", "energy"))


class TestModeRouter:

    def test_mode_switching(self):
        router = ModeRouter(mode=OperatingMode.ACTIVE)
        assert router.mode == OperatingMode.ACTIVE
        assert not router.is_listening

        # Set to PASSIVE
        router.set_mode(OperatingMode.PASSIVE)
        assert router.mode == OperatingMode.PASSIVE

        # Clean shutdown
        router.stop_passive_listening()
        assert not router.is_listening


class TestJaneTerminalChat:

    @pytest.fixture
    def chat_instance(self):
        return JaneTerminalChat(start_passive_ear=False, device="cpu")

    def test_greeting_verbalization(self, chat_instance):
        result = chat_instance.pipeline.process(text="hii")
        response = result["response_text"].lower()
        assert "hello" in response or "akhil" in response or "active" in response

    def test_math_hypothesis_inference(self, chat_instance):
        hyp = chat_instance.infer_hypothesis("what is 50 + 25")
        assert hyp is not None
        assert isinstance(hyp, MathHypothesis)
        assert "50 + 25" in hyp.lhs
        assert "75" in hyp.rhs

    def test_physics_hypothesis_inference(self, chat_instance):
        hyp = chat_instance.infer_hypothesis("input energy is 120 joules and output energy is 100 joules")
        assert hyp is not None
        assert isinstance(hyp, PhysicsHypothesis)
        assert len(hyp.quantities) == 2
        quantities = hyp.quantities
        assert quantities[0].dimension == Dimension.ENERGY
        assert quantities[0].magnitude == 120.0
        assert quantities[1].magnitude == 100.0

    def test_math_reasoning_and_verification(self, chat_instance):
        hyp = chat_instance.infer_hypothesis("calculate 30 * 4")
        result = chat_instance.pipeline.process(text="calculate 30 * 4", hypothesis=hyp)
        assert result["verification"] is not None
        assert result["verification"].is_valid
        assert result["verification"].prover_name == "Z3_SMT"
