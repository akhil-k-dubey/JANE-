"""Research Paper Ingestion & LaTeX Math Engine for Project Jane.

Fetches scientific preprints from academic sources (arXiv API, open-access repositories),
extracts LaTeX mathematical formulas, converts them to canonical AST invariants,
subjects them to the Z3 SMT Quality Gate (Karl Popper falsification),
and autonomously wires validated discoveries into Jane's 256D relational manifold.
"""

import os
import json
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from src.symbolic_verifier.latex_normalizer import LaTeXMathNormalizer
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.utils.logger import get_logger

logger = get_logger("paper_ingestor")


# Built-in offline curated scientific research papers with LaTeX formulas
# Used when workstation is offline or network is disconnected
CURATED_OFFLINE_PAPERS = [
    {
        "id": "arxiv:quant-ph/0306072",
        "title": "Macroscopic Quantum Superposition and Quantum Flux Invariants in SQUID Loops",
        "summary": (
            "We examine the macroscopic quantum behavior of superconducting loops interrupted by Josephson junctions. "
            "The magnetic flux through the loop is quantized according to the fundamental flux quantum: "
            "\\[ \\Phi_0 = \\frac{h}{2e} \\] "
            "where h denotes the Planck constant and e represents the elementary electron charge. "
            "The Josephson supercurrent phase relation satisfies: "
            "$I = I_c \\cdot \\sin(\\phi)$ "
            "demonstrating dissipationless transport under critical current bounds."
        ),
        "domain": "Superconductivity & Quantum Electronics",
        "authors": ["J. Clarke", "A. Leggett"]
    },
    {
        "id": "arxiv:astro-ph/9805201",
        "title": "Cosmological Expansion and General Relativistic Metric Curvature",
        "summary": (
            "Observations of high-redshift Type Ia supernovae demonstrate the accelerated expansion of the universe. "
            "The Friedmann dynamic equation relating scale factor acceleration to cosmological constant energy density is: "
            "\\[ \\frac{\\ddot{a}}{a} = -\\frac{4\\pi G}{3}(\\rho + \\frac{3p}{c^2}) + \\frac{\\Lambda c^2}{3} \\] "
            "under relativistic stress-energy conservation. The cosmic red-shift satisfies $1 + z = \\frac{a_0}{a}$."
        ),
        "domain": "Astrophysics & Cosmology",
        "authors": ["A. Riess", "S. Perlmutter"]
    },
    {
        "id": "arxiv:cond-mat/0509675",
        "title": "Quantum Hall Effect and Relativistic Dispersion in Graphene",
        "summary": (
            "Two-dimensional honeycomb carbon lattices exhibit relativistic Dirac fermion behavior. "
            "The energy dispersion relation is strictly linear near the Dirac points: "
            "$E = \\hbar \\cdot v_F \\cdot k$ "
            "where v_F is the Fermi velocity and k is the wavevector magnitude. "
            "The Hall conductivity quantization satisfies: "
            "\\[ \\sigma_{xy} = \\pm 4(N + \\frac{1}{2}) \\frac{e^2}{h} \\] "
            "reflecting the Berry phase of pi in massless Dirac Hamiltonian systems."
        ),
        "domain": "Condensed Matter Physics",
        "authors": ["K. S. Novoselov", "A. K. Geim"]
    },
    {
        "id": "arxiv:cs.IT/0606089",
        "title": "Compressive Sampling and Minimum Information Entropy Bounds",
        "summary": (
            "Compressed sensing enables signal reconstruction below the Nyquist-Shannon rate using sparse basis projections. "
            "The reconstruction error bound obeys the restricted isometry property: "
            "\\[ (1 - \\delta) \\|x\\|^2 \\le \\|A x\\|^2 \\le (1 + \\delta) \\|x\\|^2 \\] "
            "where delta represents the isometry constant. The information rate is bounded by Shannon capacity."
        ),
        "domain": "Information Theory & Compressed Sensing",
        "authors": ["E. Candes", "D. Donoho"]
    }
]


@dataclass
class PaperExtractionResult:
    """Extracted mathematical concepts and equations from a research paper."""
    paper_title: str
    paper_id: str
    extracted_concepts: List[Dict[str, Any]] = field(default_factory=list)
    raw_latex_blocks: List[str] = field(default_factory=list)


@dataclass
class PaperIngestionReport:
    """Summary of research paper ingestion and Z3 gate validation."""
    query: str
    papers_analyzed: int = 0
    total_formulas_found: int = 0
    accepted_concepts: List[str] = field(default_factory=list)
    rejected_violations: List[Dict[str, str]] = field(default_factory=list)
    source: str = "offline_cache"


class PaperIngestor:
    """
    Autonomous Academic Paper Ingestor for Project Jane:
    1. Searches arXiv API (with automatic fallback to curated offline academic corpus).
    2. Extracts LaTeX mathematical formulas from preprints.
    3. Normalizes LaTeX into canonical AST equations via LaTeXMathNormalizer.
    4. Proves physical and mathematical soundness via Z3 SMT Quality Gate.
    5. Natively wires accepted concepts into Jane's 256D continuous manifold.
    """

    ARXIV_API_BASE = "http://export.arxiv.org/api/query"

    def __init__(
        self,
        active_learner: ActiveLearner,
        manifold: HebbianConceptManifold,
        z3_verifier: Z3LogicVerifier,
        episodic_ledger: Optional[EpisodicLedger] = None
    ):
        self.learner = active_learner
        self.manifold = manifold
        self.verifier = z3_verifier
        self.ledger = episodic_ledger or EpisodicLedger()
        self.latex_normalizer = LaTeXMathNormalizer()

    def search_arxiv(self, query: str, max_results: int = 4) -> Tuple[List[Dict[str, Any]], str]:
        """
        Queries arXiv API for academic preprints matching the query string.
        Falls back seamlessly to local curated academic corpus if offline or network unreachable.
        """
        encoded_query = urllib.parse.quote(query)
        url = f"{self.ARXIV_API_BASE}?search_query=all:{encoded_query}&max_results={max_results}"

        papers = []
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "JaneNeuroSymbolicAgent/1.0 (academic research)"}
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)

                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled Paper"
                    summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                    paper_id = id_elem.text.strip() if id_elem is not None else ""

                    authors = []
                    for author_elem in entry.findall("atom:author", ns):
                        name_elem = author_elem.find("atom:name", ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())

                    papers.append({
                        "id": paper_id,
                        "title": title,
                        "summary": summary,
                        "authors": authors,
                        "domain": "arXiv Preprint"
                    })

                if papers:
                    return papers, "arxiv_live_api"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.info("arXiv network unreachable for query '%s' (%s); falling back to curated offline cache", query, e)
        except ET.ParseError as e:
            logger.warning("arXiv XML parsing error for query '%s': %s; falling back to offline cache", query, e)
        except Exception as e:
            logger.error("Unexpected error querying arXiv for '%s': %s", query, e, exc_info=True)

        # Offline Curated Fallback
        q_lower = query.lower()
        matched = []
        for p in CURATED_OFFLINE_PAPERS:
            text = (p["title"] + " " + p["summary"] + " " + p["domain"]).lower()
            if any(term in text for term in q_lower.split()):
                matched.append(p)

        fallback_papers = matched if matched else CURATED_OFFLINE_PAPERS[:max_results]
        return fallback_papers, "offline_curated_cache"

    def extract_paper_concepts(self, paper: Dict[str, Any]) -> PaperExtractionResult:
        """
        Extracts mathematical formulas, LaTeX blocks, and candidate concepts from a paper.
        """
        title = paper.get("title", "")
        summary = paper.get("summary", "")
        paper_id = paper.get("id", "")
        full_text = f"{title}. {summary}"

        # 1. Extract raw LaTeX blocks
        latex_blocks = self.latex_normalizer.extract_latex_blocks(full_text)

        # 2. Derive concept name candidate from paper title
        title_slug = "".join(c for c in title.lower() if c.isalnum() or c == '_' or c.isspace())
        title_words = [w for w in title_slug.split() if len(w) > 3 and w not in (
            "quantum", "macroscopic", "theory", "relativistic", "experimental", "behavior", "study", "analysis"
        )]
        if title_words:
            primary_concept = "_".join(title_words[:2])
        elif title_slug.strip():
            primary_concept = "_".join(title_slug.split()[:2])
        else:
            primary_concept = f"paper_concept_{abs(hash(title or paper_id)) % 10000}"

        # 3. Relate to existing concepts in Jane's manifold using Multi-word N-Gram tokenization
        words = []
        curr_w = []
        for ch in full_text.lower():
            if ch.isalnum():
                curr_w.append(ch)
            else:
                if curr_w:
                    words.append("".join(curr_w))
                    curr_w = []
        if curr_w:
            words.append("".join(curr_w))

        unigrams = set(words)
        bigrams = {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}
        trigrams = {f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)}
        text_ngrams = unigrams | bigrams | trigrams

        known_concepts_set = set(self.manifold.concept_names)
        matched = text_ngrams.intersection(known_concepts_set)
        related = sorted(matched)

        if not related:
            nearest = self.manifold.find_nearest_concepts_for_text(full_text, top_k=2, exclude=[primary_concept])
            if nearest:
                related = [n[0] for n in nearest]
            elif self.manifold.concept_names:
                related = list(self.manifold.concept_names[:2])
            else:
                related = []

        extracted = []
        for block in latex_blocks:
            norm_ast = self.latex_normalizer.normalize(block)
            # If the equation has relational equality or inequality
            if any(op in norm_ast for op in ("==", ">=", "<=")):
                # Determine LHS concept name if single variable
                lhs_cand = norm_ast.split("==")[0].strip() if "==" in norm_ast else norm_ast.split(">=")[0].strip()
                lhs_name = "".join(c if (c.isalnum() or c == '_') else "_" for c in lhs_cand).strip("_").lower()
                c_name = f"{primary_concept}_{lhs_name}" if len(lhs_name) > 1 else primary_concept

                extracted.append({
                    "name": c_name[:40],
                    "raw_latex": block,
                    "normalized_ast": norm_ast,
                    "related": related[:3]
                })

        return PaperExtractionResult(
            paper_title=title,
            paper_id=paper_id,
            extracted_concepts=extracted,
            raw_latex_blocks=latex_blocks
        )

    def ingest_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single paper: extracts LaTeX formulas, verifies with Z3, and wires to manifold.
        """
        extracted = self.extract_paper_concepts(paper)
        results = {
            "title": extracted.paper_title,
            "id": extracted.paper_id,
            "formulas_found": len(extracted.extracted_concepts),
            "accepted": [],
            "rejected": []
        }

        for item in extracted.extracted_concepts:
            c_name = item["name"]
            formula = item["normalized_ast"]
            related = item["related"]

            # Z3 SMT Quality Gate (Karl Popper falsification)
            is_sound, proof_msg = self.verifier.prove(formula)

            if is_sound:
                learn_msg = self.learner.learn_new_concept(
                    concept_name=c_name,
                    related_concepts=related,
                    definition_ast=formula
                )
                results["accepted"].append({
                    "concept_name": c_name,
                    "formula": formula,
                    "proof": proof_msg,
                    "log": learn_msg
                })
                # Record explicit [arXiv] provenance in EpisodicLedger
                self.ledger.record_event(
                    action_type="INGEST_ARXIV",
                    concept_name=c_name,
                    source_name=extracted.paper_id or f"arXiv:{extracted.paper_title[:30]}",
                    formula_or_axiom=formula,
                    status="COMMITTED",
                    details={"paper_title": extracted.paper_title, "proof": proof_msg}
                )
            else:
                results["rejected"].append({
                    "concept_name": c_name,
                    "formula": formula,
                    "reason": proof_msg
                })
                self.ledger.record_event(
                    action_type="INGEST_ARXIV",
                    concept_name=c_name,
                    source_name=extracted.paper_id or f"arXiv:{extracted.paper_title[:30]}",
                    formula_or_axiom=formula,
                    status="REJECTED",
                    details={"paper_title": extracted.paper_title, "reason": proof_msg}
                )

        return results

    def search_and_ingest(self, query: str, max_results: int = 3) -> PaperIngestionReport:
        """
        End-to-End Autonomous Pipeline:
        Searches academic repository for query, ingests top papers, extracts math,
        validates via Z3, and commits verified discoveries to Jane's 256D manifold.
        """
        papers, source = self.search_arxiv(query, max_results=max_results)
        report = PaperIngestionReport(
            query=query,
            papers_analyzed=len(papers),
            source=source
        )

        for p in papers:
            res = self.ingest_paper(p)
            report.total_formulas_found += res["formulas_found"]

            for acc in res["accepted"]:
                report.accepted_concepts.append(acc["concept_name"])

            for rej in res["rejected"]:
                report.rejected_violations.append(rej)

        return report
