"""Unit & Integration Tests for:
1. LaTeX Math Normalization (LaTeXMathNormalizer)
2. Academic Paper Ingestion & LaTeX Math Extraction (PaperIngestor)
3. Dynamic Knowledge Graph Verbalization (Replacing Hardcoded Keyword Branches)
"""

import pytest
import torch
from src.symbolic_verifier.latex_normalizer import LaTeXMathNormalizer
from src.core.paper_ingestor import PaperIngestor
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic
from src.translator.pretrained_translator import PretrainedLinguisticMouth
from src.core.pipeline import JanePipeline


def test_latex_normalizer_extraction_and_syntax():
    """Verifies that LaTeXMathNormalizer accurately parses mathematical environments and constructs."""
    sample_text = r"""
    The fundamental Josephson flux quantum is defined as:
    \[
    \Phi_0 = \frac{h}{2e}
    \]
    The relativistic dispersion relation satisfies:
    $E = \sqrt{p^2 c^2 + m^2 c^4}$
    while Heisenberg uncertainty imposes:
    \begin{equation}
    \Delta x \cdot \Delta p \ge \frac{\hbar}{2}
    \end{equation}
    """
    blocks = LaTeXMathNormalizer.extract_latex_blocks(sample_text)
    assert len(blocks) == 3

    # Normalize flux quantum
    norm_flux = LaTeXMathNormalizer.normalize(r"\Phi_0 = \frac{h}{2e}")
    assert "flux_quantum" in norm_flux or "phi" in norm_flux.lower()
    assert "==" in norm_flux
    assert "2 * e" in norm_flux

    # Normalize square root relativistic energy
    norm_e = LaTeXMathNormalizer.normalize(r"E = \sqrt{p^2 c^2 + m^2 c^4}")
    assert "sqrt" in norm_e
    assert "==" in norm_e
    assert "p**2" in norm_e
    assert "c**2" in norm_e

    # Normalize Heisenberg uncertainty
    norm_unc = LaTeXMathNormalizer.normalize(r"\Delta x \cdot \Delta p \ge \frac{\hbar}{2}")
    assert "delta_x" in norm_unc
    assert "delta_p" in norm_unc
    assert ">=" in norm_unc
    assert "hbar" in norm_unc


def test_paper_ingestion_with_math_extraction():
    """Verifies that PaperIngestor extracts LaTeX equations, checks with Z3, and wires concepts."""
    manifold = HebbianConceptManifold(latent_dim=256)
    z3_verifier = Z3LogicVerifier()
    self_model = SelfModelEngine()
    learner = ActiveLearner(manifold=manifold, z3_verifier=z3_verifier, self_model=self_model)
    ingestor = PaperIngestor(active_learner=learner, manifold=manifold, z3_verifier=z3_verifier)

    paper = {
        "id": "arxiv:sample/001",
        "title": "Quantum Hall Effect and Graphene Relativistic Invariants",
        "summary": "We derive the relativistic energy dispersion $E = \\hbar \\cdot v_F \\cdot k$ where v_F is the Fermi velocity.",
        "domain": "Condensed Matter Physics"
    }

    res = ingestor.ingest_paper(paper)
    assert res["formulas_found"] > 0
    assert len(res["accepted"]) > 0
    assert res["accepted"][0]["formula"] is not None
    # Verify concept was added to manifold
    assert any("graphene" in c or "hall" in c for c in manifold.concept_names)


def test_end_to_end_search_and_ingest_paper():
    """Verifies that JanePipeline.search_and_ingest_paper executes end-to-end."""
    pipeline = JanePipeline(use_pretrained_translator=True)
    initial_count = pipeline.hebbian_manifold.num_concepts

    report = pipeline.search_and_ingest_paper(query="superconductivity quantum flux", max_results=2)
    assert report.papers_analyzed > 0
    assert report.total_formulas_found > 0
    assert len(report.accepted_concepts) > 0
    assert pipeline.hebbian_manifold.num_concepts > initial_count


def test_dynamic_verbalizer_no_hardcoded_keyword_branches():
    """Verifies that verbalization is driven dynamically by the knowledge graph rather than brittle static branches."""
    self_model = SelfModelEngine()
    retriever = ManifoldRetriever(concept_names=list(self_model.data.get("knowledge_base", {}).get("domains", [])))
    critic = AntiHallucinationCritic(self_model=self_model)
    mouth = PretrainedLinguisticMouth(self_model=self_model, retriever=retriever, critic=critic)

    # 1. Quantum query: should dynamically pull from physics_principles.json
    resp_q = mouth.verbalize(
        user_prompt="what is quantum entanglement?",
        reasoned_concepts=["quantum", "entanglement"],
        proof_log=None,
        is_valid=True
    )
    assert "schrödinger" in resp_q.lower() or "superposition" in resp_q.lower() or "entanglement" in resp_q.lower()
    assert "regarding your query related to" not in resp_q.lower()

    # 2. Holographic query: should dynamically pull from physics_principles.json
    resp_h = mouth.verbalize(
        user_prompt="how does a holographic projection work?",
        reasoned_concepts=["holographic", "projection"],
        proof_log=None,
        is_valid=True
    )
    assert "wavefront" in resp_h.lower() or "interferogram" in resp_h.lower() or "beam" in resp_h.lower()

    # 3. Newly wired concept from physics principles (e.g. de_broglie_wavelength)
    resp_db = mouth.verbalize(
        user_prompt="what is the de broglie wavelength?",
        reasoned_concepts=["de_broglie_wavelength"],
        proof_log=None,
        is_valid=True
    )
    assert "wave-particle" in resp_db.lower() or "lambda" in resp_db.lower() or "momentum" in resp_db.lower()

    # 4. Epistemic gap on completely unmapped query
    resp_gap = mouth.verbalize(
        user_prompt="what is a chromodynamic skyrmion crystal?",
        reasoned_concepts=[],
        proof_log=None,
        is_valid=True
    )
    assert "not yet wired into my 256d relational manifold" in resp_gap.lower()
    assert "hallucinate" in resp_gap.lower()
