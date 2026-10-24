"""Unit & Integration Tests for Multi-Turn Rolling Memory, Companion Profile Grounding,
and Precise Self-Model (Inventory vs. Nuance) Routing.
"""

import os
import tempfile
import pytest
from src.core.chat_history import ChatHistoryManager
from src.reasoning_engine.self_model import SelfModelEngine
from src.core.intent_router import IntentRouter, RouteType
from src.core.pipeline import JanePipeline


def test_user_profile_companion_grounding():
    """Verifies that Akhil is grounded as Creator & Companion in self_model."""
    self_model = SelfModelEngine()
    profile = self_model.user_profile

    assert profile["name"] == "Akhil"
    assert "Creator" in profile["role"]
    assert "Companion" in profile["role"]
    assert "RTX 4050" in profile["environment"]

    # Test direct introspection for user identity
    assert self_model.is_self_query("who am i") is True
    assert self_model.is_self_query("do you know who am i") is True
    assert self_model.is_self_query("do you know who i am?") is True

    ans = self_model.answer_self_query("who am i")
    assert "Akhil" in ans
    assert "Creator & Companion" in ans
    assert "workstation" in ans.lower() or "rtx 4050" in ans.lower()


def test_inventory_vs_nuance_routing():
    """
    Verifies that:
    1. Quantitative count queries ('how many concepts') route to INTROSPECTIVE_SELF.
    2. Evaluative nuance queries ('do you think your knowledge graph is vast') route to CONVERSATIONAL.
    """
    router = IntentRouter()

    # Quantitative inventory queries -> INTROSPECTIVE_SELF
    q_queries = [
        "how many concepts do you currently have?",
        "how many concepts or knowledge graph do you currently have?",
        "how many graph entries do you have",
        "count your concepts",
        "what is the count of your core concepts"
    ]
    for q in q_queries:
        route, hyp = router.route(q)
        assert route == RouteType.INTROSPECTIVE_SELF, f"Failed on query: {q}"

    # Evaluative nuance queries -> CONVERSATIONAL (Must NOT be hijacked by canned inventory!)
    eval_queries = [
        "do you really think your knowledge graph is vast?",
        "is your knowledge base vast?",
        "do you think 39 concepts is vast?",
        "is your knowledge graph enough for reasoning?",
        "what is your opinion on your knowledge base?"
    ]
    for eq in eval_queries:
        route, hyp = router.route(eq)
        assert route == RouteType.CONVERSATIONAL, f"Failed on evaluative query: {eq}"


def test_evaluative_query_explains_seed_manifold():
    """Verifies that asking if the knowledge graph is vast returns a nuanced explanation, not canned stats."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    res = pipeline.process("do you really think your knowledge graph is vast?")
    assert res["route_type"] == "CONVERSATIONAL"
    resp = res["response_text"].lower()

    # Should explain that it is a focused seed manifold rather than vast
    assert "seed manifold" in resp or "not vast" in resp or "rather than a vast" in resp
    assert str(pipeline.self_model.core_concepts_count) in resp or "36" in resp or "39" in resp


def test_chat_history_manager_persistence():
    """Verifies JSONL persistence, turn appending, and reloading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "test_chat.jsonl")
        mgr = ChatHistoryManager(history_path=log_path, auto_load=True)

        assert len(mgr) == 0
        assert mgr.turn_count == 0

        # Append turn 1
        mgr.append_turn("hello", "Hello Akhil! How can I help you?")
        assert len(mgr) == 2
        assert mgr.turn_count == 1
        assert os.path.exists(log_path)

        # Append turn 2
        mgr.append_turn("what is 20 + 20", "The result of 20 + 20 is 40.")
        assert len(mgr) == 4
        assert mgr.turn_count == 2

        # Get recent history
        recent = mgr.get_recent_history(max_turns=1)
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[0]["content"] == "what is 20 + 20"
        assert recent[1]["role"] == "assistant"

        # Reload from disk into new manager
        mgr2 = ChatHistoryManager(history_path=log_path, auto_load=True)
        assert len(mgr2) == 4
        assert mgr2.turn_count == 2
        assert mgr2.messages[0]["content"] == "hello"


def test_pipeline_multi_turn_continuity():
    """Verifies that JanePipeline maintains multi-turn conversation memory across sequential calls."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")
        pipeline.chat_history = ChatHistoryManager(history_path=os.path.join(tmp_dir, "chat.jsonl"))

        # Turn 1: user asks who they are
        res1 = pipeline.process("who am i")
        assert "Akhil" in res1["response_text"]
        assert pipeline.chat_history.turn_count == 1

        # Turn 2: conversational inquiry
        res2 = pipeline.process("do know how to create a holographic projection")
        assert res2["route_type"] == "CONVERSATIONAL"
        assert pipeline.chat_history.turn_count == 2

        # Turn 3: ask about recent conversation topic
        res3 = pipeline.process("what did we talk about in last conversation")
        assert res3["route_type"] == "CONVERSATIONAL"
        assert pipeline.chat_history.turn_count == 3
        # Should reference the holographic or previous turn
        resp3 = res3["response_text"].lower()
        assert "holographic" in resp3 or "previous turn" in resp3 or "memory" in resp3
