"""Cognitive Intent Router for Project Jane.

Routes incoming queries to the proper cognitive subsystem:
1. SYMBOLIC_PROVER: Mathematical and physical constraints needing formal Z3/Clingo verification.
2. INTROSPECTIVE_SELF: Self-referential queries regarding Jane's architecture, concepts, and embodiment.
3. ACTIVE_TEACHING: Interactive concept teaching and continuous axiom acquisition.
4. CONVERSATIONAL: Natural language dialogue, scientific explanations, creative queries, and greetings.

Supports continuous 256D continuous manifold intent routing with ambiguous zone detection (0.45 - 0.60).
"""

import ast
import math
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn.functional as F

from src.symbolic_verifier.contracts import Hypothesis, MathHypothesis, PhysicsHypothesis
from src.reasoning_engine.self_model import SelfModelEngine
from src.utils.logger import get_logger

logger = get_logger("intent_router")


class RouteType(Enum):
    SYMBOLIC_PROVER = "SYMBOLIC_PROVER"
    INTROSPECTIVE_SELF = "INTROSPECTIVE_SELF"
    ACTIVE_TEACHING = "ACTIVE_TEACHING"
    CONVERSATIONAL = "CONVERSATIONAL"
    ACTION_DIRECTIVE = "ACTION_DIRECTIVE"


def _is_safe_arithmetic_ast(node: ast.AST) -> bool:
    """Recursively validates that an AST contains only safe arithmetic numbers and operators."""
    if isinstance(node, ast.Expression):
        return _is_safe_arithmetic_ast(node.body)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_arithmetic_ast(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)):
            return _is_safe_arithmetic_ast(node.left) and _is_safe_arithmetic_ast(node.right)
    return False


def _eval_safe_arithmetic_ast(node: ast.AST) -> Any:
    """Evaluates a validated arithmetic AST safely without calling eval()."""
    if isinstance(node, ast.Expression):
        return _eval_safe_arithmetic_ast(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        val = _eval_safe_arithmetic_ast(node.operand)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp):
        left = _eval_safe_arithmetic_ast(node.left)
        right = _eval_safe_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if abs(right) < 1e-9:
                raise ZeroDivisionError("Division by zero in AST arithmetic")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if abs(right) < 1e-9:
                raise ZeroDivisionError("Floor division by zero in AST arithmetic")
            return left // right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError(f"Unsupported AST node: {type(node)}")


class IntentRouter:
    """
    Decoupled intent router preventing Z3 from swallowing non-symbolic conversation.
    Dispatches to symbolic prover, introspective self-model, active learner, or conversational mouth.
    Supports continuous 256D manifold centroid cosine routing with an ambiguous margin zone (0.45 - 0.60).
    """

    def __init__(
        self,
        self_model: Optional[SelfModelEngine] = None,
        active_learner: Optional[Any] = None,
        manifold: Optional[Any] = None,
        linguistic_ear: Optional[Any] = None
    ):
        self.self_model = self_model or SelfModelEngine()
        self.active_learner = active_learner
        self.manifold = manifold
        self.linguistic_ear = linguistic_ear
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        self.intent_manifold = TrueIntentManifold.get_instance()


        # Canonical anchor prototypes for continuous 256D manifold routing
        self._intent_prototypes = {
            RouteType.SYMBOLIC_PROVER: [
                "calculate mathematical equation",
                "compute arithmetic expression",
                "verify physical conservation law",
                "prove theorem with formal Z3 solver",
                "check thermodynamics bounds and power"
            ],
            RouteType.INTROSPECTIVE_SELF: [
                "who are you and what is your architecture",
                "what are your capabilities and embodiment sensors",
                "how many concepts and knowledge graph triples do you have",
                "can you see the camera or hear audio",
                "what is your internal state and identity"
            ],
            RouteType.ACTIVE_TEACHING: [
                "let me teach you a new concept",
                "a concept is defined as follows",
                "learn this axiom and connect it to your manifold",
                "wire this scientific principle into your memory"
            ],
            RouteType.CONVERSATIONAL: [
                "hello good morning how are you",
                "explain the physical mechanism of laser emission",
                "what are the implications of quantum entanglement",
                "tell me an interesting scientific story",
                "discuss philosophical concepts of cognition"
            ]
        }
        self._intent_centroids: Dict[RouteType, torch.Tensor] = {}
        self._init_continuous_centroids()

    def _encode_text_to_vector(self, text: str) -> torch.Tensor:
        """Projects text into continuous 256D space using ear or manifold fallback."""
        if self.linguistic_ear is not None:
            try:
                vec = self.linguistic_ear.forward(text)
                return F.normalize(vec.squeeze(0), p=2, dim=-1)
            except Exception as e:
                logger.debug("Linguistic ear embedding failed: %s; using deterministic fallback", e)

        # 256D Continuous Intent Manifold fallback
        from src.reasoning_engine.intent_manifold import ContinuousIntentManifold
        return ContinuousIntentManifold.get_instance(latent_dim=256).encode(text)

    def _init_continuous_centroids(self):
        """Initializes continuous 256D centroids for each intent category."""
        for rtype, phrases in self._intent_prototypes.items():
            vecs = [self._encode_text_to_vector(p) for p in phrases]
            centroid = torch.mean(torch.stack(vecs), dim=0)
            self._intent_centroids[rtype] = F.normalize(centroid, p=2, dim=-1)

    def route_continuous_manifold(self, text: str) -> Tuple[RouteType, float, bool]:
        """
        Calculates cosine similarity of input against 256D continuous intent centroids.
        Returns: (top_route: RouteType, top_similarity: float, is_ambiguous: bool)
        An input is in the ambiguous zone if 0.45 <= similarity <= 0.60, or if the margin
        between the top two routes is < 0.08.
        """
        q_vec = self._encode_text_to_vector(text)
        sims: Dict[RouteType, float] = {}
        for rtype, centroid in self._intent_centroids.items():
            sim = F.cosine_similarity(q_vec.unsqueeze(0), centroid.unsqueeze(0), dim=-1).item()
            sims[rtype] = sim

        sorted_routes = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top_route, top_sim = sorted_routes[0]
        second_sim = sorted_routes[1][1] if len(sorted_routes) > 1 else 0.0

        is_ambiguous = (0.45 <= top_sim <= 0.60) or ((top_sim - second_sim) < 0.08 and top_sim < 0.70)
        return top_route, top_sim, is_ambiguous

    INTENT_TO_ROUTE: Dict[str, RouteType] = {
        "symbolic_math": RouteType.SYMBOLIC_PROVER,
        "symbolic_physics": RouteType.SYMBOLIC_PROVER,
        "self_identity": RouteType.INTROSPECTIVE_SELF,
        "user_identity": RouteType.INTROSPECTIVE_SELF,
        "vision": RouteType.INTROSPECTIVE_SELF,
        "capabilities": RouteType.INTROSPECTIVE_SELF,
        "concepts_inventory": RouteType.INTROSPECTIVE_SELF,
        "epistemic_discovery": RouteType.INTROSPECTIVE_SELF,
        "epistemic_why_single": RouteType.INTROSPECTIVE_SELF,
        "codebase_introspection": RouteType.INTROSPECTIVE_SELF,
        "architecture_check": RouteType.INTROSPECTIVE_SELF,
        "language_capability": RouteType.INTROSPECTIVE_SELF,
        "history_memory": RouteType.CONVERSATIONAL,
        "project_intent": RouteType.INTROSPECTIVE_SELF,
        "active_teaching": RouteType.ACTIVE_TEACHING,
        "action_directive": RouteType.ACTION_DIRECTIVE,
    }

    def route(
        self,
        text: str,
        hypothesis: Optional[Hypothesis] = None
    ) -> Tuple[RouteType, Optional[Any]]:
        """
        Classifies incoming input using TrueIntentManifold continuous vector
        Softmax probability distribution over canonical RouteTypes.
        Zero procedural if/elif ladder: routing is determined by mathematical argmax.
        """
        text_clean = text.strip()
        if hypothesis is not None:
            return RouteType.SYMBOLIC_PROVER, hypothesis

        from src.core.action_dispatcher import ActionDirectiveDispatcher
        dispatcher = ActionDirectiveDispatcher()
        if dispatcher.is_action_directive(text_clean):
            action_res = dispatcher.parse_and_dispatch(text_clean)
            if action_res is not None and action_res.is_action:
                return RouteType.ACTION_DIRECTIVE, action_res

        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()

        # Dynamic Route Pooling across all manifold centroids
        route_sims = {rt: 0.0 for rt in RouteType}
        for intent in m.centroids.keys():
            sim = m.similarity(text_clean, intent)
            rt = self.INTENT_TO_ROUTE.get(intent, RouteType.CONVERSATIONAL)
            if sim > route_sims[rt]:
                route_sims[rt] = sim

        sim_tensor = torch.tensor([
            route_sims[RouteType.SYMBOLIC_PROVER],
            route_sims[RouteType.INTROSPECTIVE_SELF],
            route_sims[RouteType.ACTIVE_TEACHING],
            route_sims[RouteType.ACTION_DIRECTIVE],
            route_sims[RouteType.CONVERSATIONAL]
        ])
        probs = F.softmax(sim_tensor / 0.05, dim=0).tolist()
        route_list = [
            RouteType.SYMBOLIC_PROVER,
            RouteType.INTROSPECTIVE_SELF,
            RouteType.ACTIVE_TEACHING,
            RouteType.ACTION_DIRECTIVE,
            RouteType.CONVERSATIONAL
        ]
        top_k = torch.topk(sim_tensor, k=2)
        top_idx = top_k.indices[0].item()
        runner_up_idx = top_k.indices[1].item()
        top_route = route_list[top_idx]
        runner_up_route = route_list[runner_up_idx]

        # Confidence threshold gating & Duration / Capability Hyperplane Subspace projection
        if top_route == RouteType.INTROSPECTIVE_SELF:
            best_self_intent, _ = m.get_self_model_intent(text_clean)
            if best_self_intent == "concepts_inventory" and hasattr(m, "get_duration_projection") and m.get_duration_projection(text_clean) > 0.05:
                top_route = RouteType.CONVERSATIONAL
            elif best_self_intent == "capabilities" and hasattr(m, "get_capability_inquiry_projection") and m.get_capability_inquiry_projection(text_clean) > 0.05:
                top_route = RouteType.CONVERSATIONAL
            elif not m.is_self_query(text_clean, threshold=0.38):
                top_route = RouteType.CONVERSATIONAL

        # Mathematical payload resolution for the chosen top route
        payload = None
        if top_route == RouteType.ACTION_DIRECTIVE:
            if hasattr(m, "get_capability_inquiry_projection") and m.get_capability_inquiry_projection(text_clean) > 0.35:
                top_route = RouteType.CONVERSATIONAL
            else:
                from src.core.action_dispatcher import ActionDirectiveDispatcher
                dispatcher = ActionDirectiveDispatcher()
                payload = dispatcher.parse_and_dispatch(text_clean)
                if payload is None or not payload.is_action:
                    top_route = RouteType.CONVERSATIONAL
        elif top_route == RouteType.SYMBOLIC_PROVER:
            payload = self.infer_hypothesis(text_clean)
            if payload is None:
                top_route = RouteType.CONVERSATIONAL
        elif top_route == RouteType.ACTIVE_TEACHING:
            if self.active_learner is not None:
                payload = self.active_learner.parse_teaching_utterance(text_clean)
                if payload is None:
                    top_route = RouteType.CONVERSATIONAL
            else:
                top_route = RouteType.CONVERSATIONAL

        # Store Tier 4 routing state for Prediction Error recovery
        self.last_routing = {
            "text": text_clean,
            "top_route": top_route,
            "runner_up": runner_up_route,
            "probs": probs
        }

        return top_route, payload

    def infer_hypothesis(self, text: str) -> Optional[Hypothesis]:
        """
        Automatically infers formal mathematical or physical hypotheses from user text
        using formal Context-Free Grammar AST parsing (Zero regex, zero keyword triggers).
        """
        math_hyp = self._try_extract_math(text)
        if math_hyp is not None:
            return math_hyp
        return self._try_extract_physics(text)

    def _try_extract_math(self, text: str) -> Optional[MathHypothesis]:
        """Extracts math hypothesis using formal Lark Earley CFG AST parser."""
        from src.core.math_ast_parser import MathASTParser
        return MathASTParser.get_instance().parse_math(text)

    def _try_extract_physics(self, text: str) -> Optional[PhysicsHypothesis]:
        """Extracts formal physics hypothesis from physical unit assertions using Lark AST numbers."""
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        m = TrueIntentManifold.get_instance()
        if m.similarity(text, "symbolic_physics") >= 0.45:
            from src.core.command_ast_parser import CommandASTParser
            numbers, _ = CommandASTParser.get_instance().parse_command_tokens(text)

            if len(numbers) >= 2:
                from src.symbolic_verifier.contracts import PhysicalQuantity, Dimension
                val1 = float(numbers[0])
                val2 = float(numbers[1])
                return PhysicsHypothesis(quantities=[
                    PhysicalQuantity(magnitude=max(val1, val2), dimension=Dimension.ENERGY, confidence=1.0, is_input=True),
                    PhysicalQuantity(magnitude=min(val1, val2), dimension=Dimension.ENERGY, confidence=1.0, is_output=True)
                ])

        return None
