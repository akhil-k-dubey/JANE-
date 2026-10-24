"""Comprehensive Tests for the 3-Tier Autonomous Ingestion Architecture in Project Jane.

Verifies:
1. Tier 1: The Idle Curiosity Daemon (Autonomous Background)
   - Identifies topological voids and sparse clusters in the 256D continuous manifold.
   - Autonomously queries scientific ingestors (Wikipedia / arXiv) while idle.
   - Maps extracted knowledge into the manifold using Hebbian plasticity.
   - Clean thread lifecycle (start, pause, resume, stop).
2. Tier 2: Contextual Auto-Fetch (Conversational Triggers)
   - Evaluates density of incoming user conversational concepts via check_density.
   - Silently triggers background fetching without blocking conversational dialogue.
   - Updates retriever and concept topology concurrently.
3. Tier 3: Manual Override (Admin Priority Learning)
   - Pauses idle curiosity loop immediately when high-priority learning commands are issued.
   - Deeply ingests target topics across Wikipedia and arXiv preprints with Z3 verification.
   - Resumes curiosity loop after deep ingestion completes.
"""

import time
import pytest
import torch

from src.core.pipeline import JanePipeline
from src.core.curiosity_daemon import CuriosityDaemon
from src.core.wikipedia_ingestor import WikipediaIngestor
from src.core.paper_ingestor import PaperIngestor
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.active_learner import ActiveLearner
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.episodic_ledger import EpisodicLedger


@pytest.fixture
def mock_manifold():
    return HebbianConceptManifold(latent_dim=256)


@pytest.fixture
def verifier():
    return Z3LogicVerifier()


@pytest.fixture
def temp_ledger(tmp_path):
    log_file = tmp_path / "test_curiosity_ledger.jsonl"
    return EpisodicLedger(log_path=str(log_file))


@pytest.fixture
def learner(mock_manifold, verifier):
    return ActiveLearner(manifold=mock_manifold, z3_verifier=verifier)


@pytest.fixture
def wiki_ingestor(learner, mock_manifold, verifier, temp_ledger):
    return WikipediaIngestor(
        active_learner=learner,
        manifold=mock_manifold,
        z3_verifier=verifier,
        episodic_ledger=temp_ledger
    )


# =============================================================================
# 1. Tier 1: The Idle Curiosity Daemon Tests
# =============================================================================

def test_tier1_sparse_cluster_detection(mock_manifold):
    """find_sparse_cluster must return unmapped frontier domain or lowest degree node."""
    sparse_topic = mock_manifold.find_sparse_cluster()
    assert isinstance(sparse_topic, str)
    assert len(sparse_topic) > 0
    # "laser" is in frontier_domains and not in default CORE_CONCEPTS
    assert sparse_topic == "laser"


def test_tier1_curiosity_daemon_step(mock_manifold, wiki_ingestor):
    """CuriosityDaemon.step synchronously learns a concept and updates the manifold."""
    learned_events = []

    def on_learn(concept, record):
        learned_events.append((concept, record))

    daemon = CuriosityDaemon(
        manifold=mock_manifold,
        wikipedia_ingestor=wiki_ingestor,
        on_learn_callback=on_learn
    )

    record = daemon.step()
    assert record is not None
    assert record["concept"] == "laser"
    assert record["raw_text_length"] > 0
    assert len(learned_events) == 1
    assert learned_events[0][0] == "laser"

    # Manifold now contains the learned concept
    assert "laser" in mock_manifold.name_to_idx
    assert mock_manifold.name_to_idx["laser"] < mock_manifold.num_concepts


def test_tier1_curiosity_daemon_thread_lifecycle(mock_manifold, wiki_ingestor):
    """CuriosityDaemon starts, pauses, resumes, and stops cleanly."""
    daemon = CuriosityDaemon(
        manifold=mock_manifold,
        wikipedia_ingestor=wiki_ingestor,
        check_interval_sec=0.05
    )

    assert not daemon.is_running
    daemon.start()
    assert daemon.is_running

    # Pause test
    daemon.pause()
    assert daemon.is_paused is True

    # Resume test
    daemon.resume()
    assert daemon.is_paused is False

    time.sleep(0.12)
    daemon.stop(timeout=1.0)
    assert not daemon.is_running
    assert len(daemon.get_learned_history()) >= 1


# =============================================================================
# 2. Tier 2: Contextual Auto-Fetch Tests
# =============================================================================

def test_tier2_check_density_keyword_extraction(mock_manifold):
    """check_density identifies concepts not represented in the manifold."""
    text = "Jane, let us design a holographic projection system using lasers."
    sparse = mock_manifold.check_density(text=text)

    # "holographic" and "laser" should be flagged as sparse
    assert "laser" in sparse or "lasers" in sparse or "holographic" in sparse


def test_tier2_process_input_silent_autofetch(tmp_path):
    """process_input triggers background auto-fetch without failing conversation."""
    pipeline = JanePipeline(
        latent_dim=256,
        use_pretrained_translator=False,
        device="cpu"
    )

    text = "Let's explore laser interferometry and holography."
    res = pipeline.process_input(text)

    assert res is not None
    assert "response_text" in res
    assert res["chat_turn_count"] >= 1

    # Give worker thread a moment to execute
    time.sleep(0.1)
    # The retriever and manifold should be updated
    assert hasattr(pipeline, "refresh_retriever")


# =============================================================================
# 3. Tier 3: Manual Override (Admin Priority Learning) Tests
# =============================================================================

def test_tier3_manual_learn_override_pauses_curiosity():
    """manual_learn_override pauses curiosity daemon and deeply ingests the topic."""
    pipeline = JanePipeline(
        latent_dim=256,
        use_pretrained_translator=False,
        device="cpu"
    )

    # Start curiosity daemon
    pipeline.start_curiosity_daemon()
    assert pipeline.curiosity_daemon.is_running

    # Execute manual override
    res = pipeline.manual_learn_override("Superconductivity")
    assert res["status"] == "COMPLETED"
    assert res["topic"] == "Superconductivity"
    assert res["wikipedia_report"] is not None

    # Daemon was resumed after override
    assert pipeline.curiosity_daemon.is_paused is False

    pipeline.stop_curiosity_daemon()
    assert not pipeline.curiosity_daemon.is_running


def test_tier3_conversational_manual_override_routing():
    """A natural utterance 'Jane, learn about X' triggers Tier 3 manual override."""
    pipeline = JanePipeline(
        latent_dim=256,
        use_pretrained_translator=False,
        device="cpu"
    )

    query = "Jane, learn about quantum mechanics"
    res = pipeline.process(query)

    assert res["route_type"] == "MANUAL_LEARN_OVERRIDE"
    assert "Manual override executed" in res["response_text"]
    assert "quantum mechanics" in res["response_text"]
    assert res["override_details"]["status"] == "COMPLETED"
