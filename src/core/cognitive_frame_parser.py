"""Cognitive Frame Parser for Project Jane.

Translates fluid human natural language into structured, executable Cognitive Frames
(Action DAG nodes and State Queries) using continuous 256D vector semantic projection.
Zero procedural keyword lists, substring in lower scans, or regex patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.reasoning_engine.intent_manifold import ContinuousIntentManifold


@dataclass
class CognitiveFrame:
    frame_type: str  # "COMMAND_INGEST", "QUERY_TASK_PROGRESS", "CAPABILITY_CHALLENGE", "INTROSPECTIVE_SELF", "CONVERSATIONAL"
    action: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class CognitiveFrameParser:
    """
    Decoupled Frame & Argument Extractor.
    Combines continuous 256D intent projection with dynamic semantic slot resolution.
    Eliminates all hardcoded keyword lists and procedural if/elif substring scans.
    """

    def __init__(self, intent_manifold: Optional[ContinuousIntentManifold] = None):
        self.intent_manifold = intent_manifold or ContinuousIntentManifold.get_instance()

    def parse(self, text: str, has_active_task: bool = False) -> CognitiveFrame:
        """
        Parses raw text into an executable CognitiveFrame using continuous vector resonance.
        """
        clean = text.strip()

        # Compute continuous cosine resonance across action centroids
        sim_progress = self.intent_manifold.similarity(clean, "query_task_progress")
        sim_ingest = self.intent_manifold.similarity(clean, "command_ingest")
        sim_challenge = self.intent_manifold.similarity(clean, "capability_challenge")

        is_self = self.intent_manifold.is_self_query(clean)
        best_self_intent, best_self_sim = self.intent_manifold.get_self_model_intent(clean)

        # 1. Active Task Progress / Telemetry Probing
        progress_threshold = 0.50 if has_active_task else 0.65
        if sim_progress >= progress_threshold and sim_progress > sim_ingest and sim_progress > sim_challenge and not (is_self and best_self_sim > sim_progress):
            return CognitiveFrame(
                frame_type="QUERY_TASK_PROGRESS",
                action="GET_TELEMETRY",
                parameters={"query": text},
                confidence=sim_progress
            )

        # 2. Ingestion & Expansion Command (Path 3: Mood Hyperplane Subspace + Lark CFG Parameter Extraction)
        mood_lambda = self.intent_manifold.get_mood_projection(clean)
        sim_inquiry = self.intent_manifold.similarity(clean, "interrogative_inquiry")
        sim_repair = self.intent_manifold.similarity(clean, "conversational_repair")
        sim_eval = self.intent_manifold.similarity(clean, "evaluative_scale")
        is_imperative = (mood_lambda <= -0.10)

        if (
            is_imperative
            and sim_ingest >= 0.45
            and sim_ingest > sim_challenge
            and sim_ingest > sim_inquiry
            and sim_ingest > sim_repair
            and sim_ingest > sim_eval
            and not (is_self and best_self_sim > sim_ingest)
        ):
            # Formal Lark Context-Free Grammar AST Token Extraction (Zero .split() / .isdigit())
            from src.core.command_ast_parser import CommandASTParser
            ast_ints, ast_words = CommandASTParser.get_instance().parse_command_tokens(clean)
            target_count = max(ast_ints) if ast_ints else 50

            # Source resolution via continuous vector similarity against source concepts
            sim_arxiv = self.intent_manifold.similarity(clean, "arxiv")
            sim_wiki = self.intent_manifold.similarity(clean, "wikipedia")
            source = "arxiv" if sim_arxiv > sim_wiki and sim_arxiv >= 0.40 else "wikipedia"

            # Dynamic continuous domain classification
            from src.reasoning_engine.domain_topology import DomainTopologyEngine
            dom_engine = DomainTopologyEngine.get_instance()
            domain_name, _ = dom_engine.classify_domain(clean)

            canonical_map = {
                "Physics & Energy": "physics",
                "Computer Science & Coding": "computer_science",
                "Mathematics & Logic": "mathematics",
                "Cognitive & Epistemic": "cognitive",
                "Speech & Audio Modality": "speech_modality"
            }
            domain = canonical_map.get(domain_name, domain_name.lower().replace(" & ", "_").replace(" ", "_"))

            return CognitiveFrame(
                frame_type="COMMAND_INGEST",
                action="SPAWN_INGESTION_TASK",
                parameters={
                    "source": source,
                    "target_count": target_count,
                    "domain": domain,
                    "raw_text": text,
                    "ast_ints": ast_ints,
                    "ast_words": ast_words
                },
                confidence=sim_ingest
            )

        # 3. Capability Challenge / Hypothesis Generation
        if sim_challenge >= 0.60 and not (is_self and best_self_sim > sim_challenge):
            return CognitiveFrame(
                frame_type="CAPABILITY_CHALLENGE",
                action="EXECUTE_HYPOTHESIS",
                parameters={"query": text},
                confidence=sim_challenge
            )

        # 4. Introspective Self Queries
        if is_self:
            if best_self_intent == "language_capability":
                return CognitiveFrame(
                    frame_type="INTROSPECTIVE_SELF",
                    action="EXPLAIN_LANGUAGE_CAPABILITY",
                    parameters={"best_intent": "language_capability"},
                    confidence=best_self_sim
                )
            return CognitiveFrame(
                frame_type="INTROSPECTIVE_SELF",
                action="GET_SELF_TELEMETRY",
                parameters={"best_intent": best_self_intent, "similarity": best_self_sim},
                confidence=best_self_sim
            )

        # 5. Default: Pure Conversational Dialogue
        return CognitiveFrame(
            frame_type="CONVERSATIONAL",
            action="VERBALIZE",
            parameters={"query": text},
            confidence=0.85
        )
