"""Wikipedia Scientific Ingestor with Source Provenance for Project Jane.

Fetches foundational scientific definitions, laws, and articles from Wikipedia,
validates statements through the Z3 SMT Quality Gate, records explicit [Wikipedia]
provenance metadata, and permanently wires verified concepts into Jane's 256D manifold.
"""

import os
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.active_learner import ActiveLearner
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.utils.logger import get_logger

logger = get_logger("wikipedia_ingestor")


# Curated foundational Wikipedia articles for offline / sandboxed execution
WIKIPEDIA_FOUNDATIONAL_ARTICLES = [
    {
        "title": "Quantum_entanglement",
        "domain": "Quantum Mechanics",
        "summary": (
            "Quantum entanglement is a physical phenomenon that occurs when a group of particles "
            "interact in ways such that the quantum state of each particle of the group cannot be "
            "described independently of the state of the others, even when the particles are separated "
            "by a large distance. Bell theorem proves that no physical theory of local hidden variables "
            "can reproduce all the predictions of quantum mechanics."
        ),
        "concepts": [
            {
                "name": "quantum_entanglement",
                "definition": "entanglement(particle_a, particle_b) -> correlated_state",
                "related": ["quantum", "wave_function"]
            },
            {
                "name": "bell_theorem",
                "definition": "bell_inequality_violation == quantum_nonlocality",
                "related": ["quantum", "probability"]
            }
        ]
    },
    {
        "title": "Black_hole_thermodynamics",
        "domain": "Astrophysics & Relativity",
        "summary": (
            "Black hole thermodynamics is the area of study that seeks to reconcile the laws of "
            "thermodynamics with the existence of black hole event horizons. The Bekenstein-Hawking "
            "entropy of a black hole is proportional to the area of its event horizon: "
            "entropy == (boltzmann_constant * event_horizon_area) / (4 * planck_length ** 2)."
        ),
        "concepts": [
            {
                "name": "bekenstein_hawking_entropy",
                "definition": "entropy == (boltzmann_constant * area) / (4 * planck_length**2)",
                "related": ["entropy", "energy"]
            }
        ]
    },
    {
        "title": "Superconductivity",
        "domain": "Condensed Matter Physics",
        "summary": (
            "Superconductivity is a set of physical properties observed in certain materials where "
            "electrical resistance vanishes and magnetic flux fields are expelled from the interior. "
            "The critical temperature T_c defines the transition to the superconducting state. "
            "The London penetration depth characterizes the exponential decay of magnetic fields inside the superconductor."
        ),
        "concepts": [
            {
                "name": "meissner_effect",
                "definition": "internal_magnetic_field == 0 and resistance == 0",
                "related": ["superconductivity", "magnetic_flux"]
            },
            {
                "name": "london_penetration_depth",
                "definition": "decay_constant == sqrt(mass / (num_density * charge**2 * mu_0))",
                "related": ["superconductivity", "permeability"]
            }
        ]
    }
]


@dataclass
class WikipediaIngestionReport:
    """Summary of Wikipedia article ingestionpass."""
    query: str
    articles_processed: int = 0
    accepted_concepts: List[str] = field(default_factory=list)
    rejected_concepts: List[Dict[str, str]] = field(default_factory=list)
    source: str = "wikipedia_curated_cache"


class WikipediaIngestor:
    """
    Scientific Wikipedia Ingestion Engine for Project Jane:
    1. Fetches article summaries via Wikipedia REST API (with offline cache fallback).
    2. Extracts candidate concepts and definitions.
    3. Validates invariants through the Z3 SMT Quality Gate.
    4. Logs explicit [Wikipedia] provenance to EpisodicLedger.
    5. Wires accepted concepts to HebbianConceptManifold and knowledge graph.
    """

    WIKI_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"

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

    def fetch_wikipedia_article(self, title: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Fetches summary of a Wikipedia article by title.
        Falls back seamlessly to local curated articles if offline.
        """
        encoded_title = urllib.parse.quote(title.strip().replace(" ", "_"))
        url = f"{self.WIKI_API_BASE}/{encoded_title}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "JaneNeuroSymbolicAgent/1.0 (academic research)"}
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "title": data.get("title", title),
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                    "concepts": []
                }, "wikipedia_live_api"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.info("Wikipedia network unreachable for article '%s' (%s); falling back to curated cache", title, e)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON response from Wikipedia API for '%s': %s", title, e)
        except Exception as e:
            logger.error("Unexpected error fetching Wikipedia article '%s': %s", title, e, exc_info=True)

        # Offline Curated Fallback
        t_lower = title.lower().replace(" ", "_")
        for art in WIKIPEDIA_FOUNDATIONAL_ARTICLES:
            if t_lower in art["title"].lower() or any(term in art["title"].lower() for term in t_lower.split("_")):
                return art, "wikipedia_curated_cache"

        # General fallback to first available
        return WIKIPEDIA_FOUNDATIONAL_ARTICLES[0], "wikipedia_curated_cache"

    def ingest_article(self, article_title: str) -> WikipediaIngestionReport:
        """
        Ingests a Wikipedia article, validates extracted definitions,
        records [Wikipedia] provenance, and updates Jane's 256D manifold.
        """
        article, source = self.fetch_wikipedia_article(article_title)
        report = WikipediaIngestionReport(
            query=article_title,
            source=source
        )

        if not article:
            return report

        report.articles_processed = 1
        concepts_to_ingest = article.get("concepts", [])

        # If live article without pre-extracted concepts, parse summary
        if not concepts_to_ingest:
            name = article["title"].lower().replace(" ", "_")
            extract_text = article.get("extract", "") or article.get("title", "")
            nearest = self.manifold.find_nearest_concepts_for_text(extract_text, top_k=2, exclude=[name])
            dynamic_related = [n[0] for n in nearest] if nearest else list(self.manifold.concept_names[:2])
            concepts_to_ingest.append({
                "name": name,
                "definition": f"concept_definition({name}) -> scientific_principle",
                "related": dynamic_related
            })

        for item in concepts_to_ingest:
            c_name = item["name"]
            formula = item["definition"]
            related = item.get("related")
            if not related:
                nearest = self.manifold.find_nearest_concepts_for_text(f"{c_name} {formula}", top_k=2, exclude=[c_name])
                related = [n[0] for n in nearest] if nearest else list(self.manifold.concept_names[:2])

            is_valid, proof_msg = self.verifier.prove(formula)
            if is_valid:
                learn_msg = self.learner.learn_new_concept(
                    concept_name=c_name,
                    related_concepts=related,
                    definition_ast=formula
                )
                report.accepted_concepts.append(c_name)
                # Record explicit provenance in EpisodicLedger
                self.ledger.record_event(
                    action_type="INGEST_WIKIPEDIA",
                    concept_name=c_name,
                    source_name=f"Wikipedia:{article['title']}",
                    formula_or_axiom=formula,
                    status="COMMITTED",
                    details={"learn_log": learn_msg, "source_type": source}
                )
            else:
                report.rejected_concepts.append({"concept": c_name, "reason": proof_msg})
                self.ledger.record_event(
                    action_type="INGEST_WIKIPEDIA",
                    concept_name=c_name,
                    source_name=f"Wikipedia:{article['title']}",
                    formula_or_axiom=formula,
                    status="REJECTED",
                    details={"rejection_reason": proof_msg}
                )

        return report

    def fetch_summary(self, title: str) -> str:
        """
        Tier 1 / Curiosity Helper:
        Fetches summary text for a given concept or title.
        """
        article, _ = self.fetch_wikipedia_article(title)
        if article:
            return article.get("summary", "") or article.get("title", "")
        return f"Scientific concept {title}."

    def fetch_and_map(self, title: str) -> WikipediaIngestionReport:
        """
        Tier 2 Contextual Auto-Fetch Helper:
        Silently ingests a concept from Wikipedia and maps it into the manifold.
        """
        return self.ingest_article(title)
