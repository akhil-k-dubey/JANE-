"""Unit tests for Pragmatic Grounder and Pragmatic Intent Abduction in Project Jane.

Verifies:
1. Pragmatic clarification parsing ("it's my way of saying that - you are working properly", "I meant, it's good").
2. Zero epistemic gap false positives on casual dialogue, praise, banter, and idioms.
3. Multi-turn idiom triangulation and persistent personal lexicon storage.
4. Stoic, JARVIS-style acknowledgement and state calibration.
5. Zero regular expression constraint.
"""

import os
import shutil
import pytest
from src.reasoning_engine.pragmatic_grounder import PragmaticGrounder, LearnedUserIdiom
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.translator.pretrained_translator import PretrainedLinguisticMouth
from src.reasoning_engine.self_model import SelfModelEngine


@pytest.fixture
def temp_pragmatics_path(tmp_path):
    return str(tmp_path / "user_pragmatics.json")


@pytest.fixture
def temp_ledger(tmp_path):
    log_file = str(tmp_path / "episodic_ledger.jsonl")
    return EpisodicLedger(log_path=log_file)


def test_pragmatic_clarification_parsing(temp_pragmatics_path, temp_ledger):
    grounder = PragmaticGrounder(pragmatics_path=temp_pragmatics_path, episodic_ledger=temp_ledger)

    # 1. Parse clarification with prior history
    history = [
        {"role": "user", "content": "alr jane , i see you have mastered your modifications"},
        {"role": "assistant", "content": "Understood."}
    ]
    res1 = grounder.parse_pragmatic_clarification(
        "it is just a praising pharse. I meant , it's good",
        history=history
    )
    assert res1.is_clarification is True
    assert "good" in res1.semantic_meaning or "praising" in res1.expression
    assert res1.canonical_intent in ("system_status_operational", "affirmation_praise", "pleasantry")
    assert "Understood, Akhil" in res1.acknowledgement_text

    # 2. Parse direct idiom explanation
    res2 = grounder.parse_pragmatic_clarification(
        "it's my way of saying that - you are working properly",
        history=history
    )
    assert res2.is_clarification is True
    assert "working properly" in res2.semantic_meaning
    assert res2.canonical_intent == "system_status_operational"
    assert "Understood, Akhil" in res2.acknowledgement_text

    # Verify persistent storage
    assert os.path.exists(temp_pragmatics_path)
    reloaded = PragmaticGrounder(pragmatics_path=temp_pragmatics_path)
    assert len(reloaded.idioms) > 0


def test_idiom_matching(temp_pragmatics_path):
    grounder = PragmaticGrounder(pragmatics_path=temp_pragmatics_path)
    grounder.parse_pragmatic_clarification(
        "it's my way of saying that - you are working properly",
        history=[{"role": "user", "content": "mastered modifications"}]
    )

    # Match exact or token overlap
    matched = grounder.match_user_idiom("mastered modifications")
    assert matched is not None
    assert matched.canonical_intent == "system_status_operational"


def test_manifold_retriever_no_false_epistemic_gaps():
    """Verifies that casual English, praise, and dialogue do NOT trigger epistemic gaps."""
    retriever = ManifoldRetriever(concept_names=["mass", "velocity", "force", "acceleration", "energy", "entropy"])

    # Genuine technical inquiry -> Epistemic gap
    ctx_gap = retriever.retrieve("what is a topological insulator?")
    assert ctx_gap.is_epistemic_gap is True
    assert "topological insulator" in ctx_gap.unmapped_concept

    # Casual praise -> NOT an epistemic gap
    ctx1 = retriever.retrieve("alr jane , i see you have mastered your modifications")
    assert ctx1.is_epistemic_gap is False
    assert ctx1.unmapped_concept is None

    # Clarification remark -> NOT an epistemic gap
    ctx2 = retriever.retrieve("it is just a praising pharse. I meant , it's good")
    assert ctx2.is_epistemic_gap is False
    assert ctx2.unmapped_concept is None

    # Laughter and compliment -> NOT an epistemic gap
    ctx3 = retriever.retrieve("hahaha, i love it. You are awesome")
    assert ctx3.is_epistemic_gap is False
    assert ctx3.unmapped_concept is None

    # User idiom explanation -> NOT an epistemic gap
    ctx4 = retriever.retrieve("it's my way of saying that - you are working properly")
    assert ctx4.is_epistemic_gap is False
    assert ctx4.unmapped_concept is None


def test_pretrained_linguistic_mouth_handles_praise_without_rejection():
    """Verifies that PretrainedLinguisticMouth does NOT output robotic manifold rejections for conversational praise."""
    mouth = PretrainedLinguisticMouth()

    # User explains praise
    resp = mouth.verbalize(
        user_prompt="it's my way of saying that - you are working properly",
        reasoned_concepts=[],
        history=[{"role": "user", "content": "alr jane , i see you have mastered your modifications"}]
    )

    # Must NOT contain the robotic manifold gap rejection
    assert "not yet wired into my 256D relational manifold" not in resp
    assert "Understood, Akhil" in resp
