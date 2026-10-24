"""Symbolic Verifier Orchestrator: Formal Type Dispatch & Tiered Spatial Architecture."""

from typing import Dict, Type, Callable, Any
from .contracts import (
    Hypothesis,
    MathHypothesis,
    PhysicsHypothesis,
    SpatialHypothesis,
    SpatialLogicHypothesis,
    KinematicTrajectoryHypothesis,
    VerificationResult
)
from .z3_interface import Z3LogicVerifier
from .clingo_interface import ClingoSpatialVerifier
from .kinematic_interface import ContinuousKinematicsVerifier


class SymbolicVerifierOrchestrator:
    """
    Formal Type-Dispatched Verifier.
    Routes formal hypothesis dataclass objects directly to registered domain provers
    using Python's native type system without string matching or regex.
    """

    def __init__(self):
        self.z3_verifier = Z3LogicVerifier()
        self.clingo_verifier = ClingoSpatialVerifier()
        self.kinematic_verifier = ContinuousKinematicsVerifier()

        # Type-based registry: Maps Hypothesis class to the corresponding prover method
        self.registry: Dict[Type[Hypothesis], Callable[[Any], VerificationResult]] = {
            MathHypothesis: self.z3_verifier.verify_math,
            PhysicsHypothesis: self.z3_verifier.verify_physics,
            SpatialLogicHypothesis: self.clingo_verifier.verify_logic_permissions,
            KinematicTrajectoryHypothesis: self.kinematic_verifier.verify_trajectory,
            SpatialHypothesis: self.clingo_verifier.verify_spatial
        }

    def register_prover(self, hypothesis_cls: Type[Hypothesis], prover_fn: Callable[[Any], VerificationResult]):
        """Allows dynamic dependency injection of new domain provers."""
        self.registry[hypothesis_cls] = prover_fn

    def verify(self, hypothesis: Hypothesis) -> VerificationResult:
        """
        Dispatches hypothesis verification based on its formal Python type.
        """
        hyp_type = type(hypothesis)
        prover = self.registry.get(hyp_type)

        if not prover:
            raise TypeError(f"No symbolic prover registered for hypothesis type: {hyp_type}")

        return prover(hypothesis)
