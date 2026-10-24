"""Unit & Integration Tests for Conversational Router and Introspective Self-Model."""

import pytest
from src.core.intent_router import IntentRouter, RouteType
from src.reasoning_engine.self_model import SelfModelEngine
from src.core.pipeline import JanePipeline


def test_self_model_introspective_answers():
    self_model = SelfModelEngine()

    assert self_model.core_concepts_count >= 36
    assert self_model.kg_entries_count >= 101
    assert "Physics & Energy" in self_model.domains

    # 1. Concept count query
    ans_concepts = self_model.answer_self_query("how many concepts or knowledge graph do you currently have?")
    assert str(self_model.core_concepts_count) in ans_concepts
    assert str(self_model.kg_entries_count) in ans_concepts
    assert "Physics & Energy" in ans_concepts

    # 2. Screen / camera query
    ans_screen = self_model.answer_self_query("can you see the scree jane?")
    assert "cannot see your screen" in ans_screen.lower()
    assert "terminal" in ans_screen.lower()

    # 3. Identity query
    ans_who = self_model.answer_self_query("who are you")
    assert "Jane" in ans_who
    assert "256D" in ans_who


def test_intent_router_classification():
    router = IntentRouter()

    # Introspective queries
    r1, h1 = router.route("how many concepts or knowledge graph do you currently have?")
    assert r1 == RouteType.INTROSPECTIVE_SELF
    assert h1 is None

    r2, h2 = router.route("can you see the scree jane?")
    assert r2 == RouteType.INTROSPECTIVE_SELF
    assert h2 is None

    r3, h3 = router.route("who are you")
    assert r3 == RouteType.INTROSPECTIVE_SELF
    assert h3 is None

    # Conversational / Explanatory queries
    r4, h4 = router.route("do know how to create a holographic projection")
    assert r4 == RouteType.CONVERSATIONAL
    assert h4 is None

    r5, h5 = router.route("hello there")
    assert r5 == RouteType.CONVERSATIONAL
    assert h5 is None

    # Symbolic prover math / physics tasks
    r6, h6 = router.route("calculate 25 * 4")
    assert r6 == RouteType.SYMBOLIC_PROVER
    assert h6 is not None

    r7, h7 = router.route("is 50 + 50 == 100")
    assert r7 == RouteType.SYMBOLIC_PROVER
    assert h7 is not None


def test_pipeline_conversational_bypasses_z3():
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    # 1. Conversational explanation query: Z3 must NOT swallow it!
    res_holo = pipeline.process("do know how to create a holographic projection")
    assert res_holo["route_type"] == "CONVERSATIONAL"
    assert res_holo["verification"] is None
    # Must NOT have the robotic "All physical and mathematical constraints verified" boilerplate!
    assert "all physical and mathematical constraints verified" not in res_holo["response_text"].lower()
    assert any(w in res_holo["response_text"].lower() for w in ("laser", "beam", "holographic", "interference", "wavefront"))

    # 2. Introspective self-model query: Returns exact concept counts
    res_concepts = pipeline.process("how many concepts or knowledge graph do you currently have?")
    assert res_concepts["route_type"] == "INTROSPECTIVE_SELF"
    assert res_concepts["verification"] is None
    assert str(pipeline.self_model.core_concepts_count) in res_concepts["response_text"]
    assert str(pipeline.self_model.kg_entries_count) in res_concepts["response_text"]

    # 3. Screen visibility query: Answers honestly about CLI embodiment
    res_screen = pipeline.process("can you see the scree jane?")
    assert res_screen["route_type"] == "INTROSPECTIVE_SELF"
    assert res_screen["verification"] is None
    assert "cannot see your screen" in res_screen["response_text"].lower()

    # 4. "Why only" query about discoveries explains Karl Popper cycles
    res_why = pipeline.process("why you only show force work bridge as a discovered concept")
    assert res_why["route_type"] == "INTROSPECTIVE_SELF"
    assert "single cycle" in res_why["response_text"].lower() or "popperian" in res_why["response_text"].lower()
    assert "force_work_bridge" in res_why["response_text"]
    assert not res_why["response_text"].startswith("Yes.")

    # 4.5. Dynamic Concept Names Inventory (e.g. "what concepts you know right now")
    res_known = pipeline.process("what concepts you know right now")
    assert res_known["route_type"] == "INTROSPECTIVE_SELF"
    assert any(c in res_known["response_text"] for c in ("energy", "force", "mass", "momentum"))
    assert "You can ask me to solve calculations" not in res_known["response_text"]

    res_list = pipeline.process("list your concepts")
    assert res_list["route_type"] == "INTROSPECTIVE_SELF"
    assert any(c in res_list["response_text"] for c in ("energy", "force", "mass", "momentum"))

    # 4. Math calculation: Formally proven by Z3 SMT prover!
    res_math = pipeline.process("calculate 15 + 25")
    assert res_math["route_type"] == "SYMBOLIC_PROVER"
    assert res_math["verification"] is not None
    assert res_math["verification"].is_valid is True
    assert res_math["hypothesis"] is not None
    assert res_math["hypothesis"].rhs == "40"


def test_continuous_manifold_intent_routing():
    """
    Verifies 100% Manifold-Driven 256D Vector Intent Routing:
    - Synonym Invariance ('display buffer' -> 'vision')
    - Typo Resilience ('screee', 'camerra' -> 'vision')
    - Phrasing Invariance ('Who is sitting at the terminal?' -> 'user_identity')
    - Disjoint Subspaces (Poincaré geometry does not collide with self identity)
    - Zero brittle regex or string tuple matching.
    """
    from src.reasoning_engine.intent_manifold import ContinuousIntentManifold
    m = ContinuousIntentManifold.get_instance(latent_dim=256)

    # 1. Synonym Invariance
    sim_disp = m.similarity("Can you inspect my display buffer?", "vision")
    assert sim_disp >= 0.72, f"Expected vision sim >= 0.72, got {sim_disp}"

    # 2. Typo Resilience via subword character n-grams
    sim_screee = m.similarity("screee", "vision")
    assert sim_screee >= 0.72, f"Expected typo screee sim >= 0.72, got {sim_screee}"

    sim_cam = m.similarity("camerra feed", "vision")
    assert sim_cam >= 0.72, f"Expected typo camerra sim >= 0.72, got {sim_cam}"

    # 3. Phrasing Invariance for User Identity
    sim_user1 = m.similarity("Who is sitting at the terminal?", "user_identity")
    assert sim_user1 >= 0.72, f"Expected user_identity sim >= 0.72, got {sim_user1}"

    sim_user2 = m.similarity("Do you recognize the user?", "user_identity")
    assert sim_user2 >= 0.72, f"Expected user_identity sim >= 0.72, got {sim_user2}"

    # 4. Epistemic Discovery Ingestion
    sim_disc = m.similarity("What new physical principles were derived recently?", "epistemic_discovery")
    assert sim_disc >= 0.72, f"Expected epistemic_discovery sim >= 0.72, got {sim_disc}"

    # 5. Evaluative Scale Nuance
    sim_eval = m.similarity("is your knowledge vast or enough?", "evaluative_scale")
    assert sim_eval >= 0.72, f"Expected evaluative_scale sim >= 0.72, got {sim_eval}"

    # 6. Disjoint Subspaces: Mathematical Poincaré model does not collide with self identity
    sim_geom = m.similarity("Poincare disk model in non-euclidean geometry", "geometry_math")
    sim_self = m.similarity("Poincare disk model in non-euclidean geometry", "self_identity")
    assert sim_geom >= 0.80, f"Expected geometry sim >= 0.80, got {sim_geom}"
    assert sim_self < 0.40, f"Expected self sim < 0.40, got {sim_self}"

    # 7. End-to-end SelfModelEngine verification
    sm = SelfModelEngine()
    assert sm.is_self_query("Can you inspect my display buffer?")
    assert sm.is_self_query("Who is sitting at the terminal?")
    assert sm.is_self_query("Do you recognize the user?")
    assert not sm.is_self_query("Poincare disk model in non-euclidean geometry")
    assert not sm.is_self_query("is your knowledge vast or enough?")  # Evaluative nuance routes to mouth


def test_capability_inquiry_hyperplane_routing():
    """
    Verifies that asking if Jane can code or perform a task routes to conversational mouth,
    while requesting an architectural list routes to introspective self-model.
    """
    router = IntentRouter()
    
    # Inquiries about capability should route to conversational mouth
    inquiries = [
        "can you write code",
        "do you fully know python programming? are you fluent in writting it ?",
        "i want to if you can write code in python",
        "can you or can you not code"
    ]
    for q in inquiries:
        r, _ = router.route(q)
        assert r == RouteType.CONVERSATIONAL, f"Expected {q} to route to CONVERSATIONAL, got {r}"

    # Architectural blueprint / list requests should route to introspective self
    list_queries = [
        "what are your capabilities",
        "list your capabilities",
        "tell me all your core capabilities"
    ]
    for q in list_queries:
        r, _ = router.route(q)
        assert r == RouteType.INTROSPECTIVE_SELF, f"Expected {q} to route to INTROSPECTIVE_SELF, got {r}"


def test_genesis_anchor_priority_shift_and_resurfacer_suppression(tmp_path):
    """
    Verifies that project priority shifts deactivate old anchors,
    and SemanticResurfacer only evaluates active, non-suppressed anchors.
    """
    from src.reasoning_engine.genesis_node import GenesisNodeRegistry, GenesisAnchor
    from src.reasoning_engine.semantic_resurfacer import SemanticResurfacer
    import torch
    import time

    # Create temporary ledger
    ledger_file = tmp_path / "genesis_nodes.json"
    registry = GenesisNodeRegistry()
    registry.LEDGER_PATH = str(ledger_file)
    registry.anchors = []

    # 1. Project initiation
    a1 = registry.maybe_create_anchor("Jane, we are starting a project to redesign the reactor cooling system.")
    assert a1 is not None
    assert a1.is_active is True
    assert a1.status == "active"
    assert len(registry.get_active_anchors()) == 1

    # 2. Priority shift
    shift_text = "well change the project priority first that we going to make now is holographic projection , reactor cooling comes after that. alright?"
    assert registry.is_priority_shift_utterance(shift_text) is True
    a2 = registry.maybe_create_anchor(shift_text)
    assert a2 is not None
    assert a2.is_active is True
    assert a2.status == "active"
    
    # Old anchor must be superseded and not in active anchors
    assert a1.is_active is False
    assert a1.status == "superseded"
    active_anchors = registry.get_active_anchors()
    assert len(active_anchors) == 1
    assert active_anchors[0].anchor_id == a2.anchor_id

    # 3. Resurfacer ignores inactive and suppressed anchors
    alerts = []
    def alert_cb(sim, anchor):
        alerts.append((sim, anchor.anchor_id))

    # Mock state returning a vector diverging from both anchors
    divergent_vec = torch.zeros(256)
    resurfacer = SemanticResurfacer(
        genesis_registry=registry,
        get_current_state_vec=lambda: divergent_vec,
        alert_callback=alert_cb
    )

    # Force created_at to be older than 300s
    a1.created_at = time.time() - 400
    a2.created_at = time.time() - 400

    # Test alignment check: a1 should NEVER trigger alert because it is superseded
    resurfacer._check_alignment()
    fired_ids = [aid for _, aid in alerts]
    assert a1.anchor_id not in fired_ids

    # Suppress a2 as well
    resurfacer.suppress_anchor(a2.anchor_id)
    alerts.clear()
    resurfacer._check_alignment()
    assert len(alerts) == 0

    # Active override test
    resurfacer.unsuppress_anchor(a2.anchor_id)
    a2.is_active = True
    a2.status = "active"
    a2.last_activated = 0.0
    resurfacer.set_active_override(True)
    alerts.clear()
    resurfacer._check_alignment()
    assert len(alerts) == 0


def test_project_and_history_inquiries_do_not_create_anchors(tmp_path):
    """
    Verifies that questions asking about projects or conversation history
    route to INTROSPECTIVE_SELF and NEVER create spurious Genesis Anchors.
    """
    from src.reasoning_engine.genesis_node import GenesisNodeRegistry
    from src.core.intent_router import IntentRouter, RouteType

    ledger_file = tmp_path / "genesis_nodes.json"
    registry = GenesisNodeRegistry()
    registry.LEDGER_PATH = str(ledger_file)
    registry.anchors = []

    # Inquiries must be recognized as project inquiries
    assert registry.is_project_inquiry("what was our project") is True
    assert registry.is_project_inquiry("what was our last project") is True
    assert registry.is_project_inquiry("can you show me history of our convo") is True
    assert registry.is_project_inquiry("tell me about our project") is True
    assert registry.is_project_inquiry("what is our current project") is True

    # Real priority shift declarations must NOT be flagged as inquiries
    assert registry.is_project_inquiry("change the project priority to holographic projection") is False
    assert registry.is_project_inquiry("we are starting a project to redesign the reactor") is False

    # Inquiries must NEVER create anchors
    assert registry.maybe_create_anchor("what was our project") is None
    assert registry.maybe_create_anchor("what was our last project") is None
    assert registry.maybe_create_anchor("can you show me history of our convo") is None
    assert len(registry.anchors) == 0

    # Route check: inquiries route to INTROSPECTIVE_SELF
    router = IntentRouter()
    r1, _ = router.route("what was our project")
    assert r1 == RouteType.INTROSPECTIVE_SELF

    r2, _ = router.route("can you show me history of our convo")
    assert r2 == RouteType.CONVERSATIONAL



