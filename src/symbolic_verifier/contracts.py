from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union


class Hypothesis:
    """Base class for all verifiable neural hypotheses."""
    pass


class Dimension(Enum):
    """Formal SI base dimensions. Raw strings are strictly prohibited."""
    MASS = "M"
    LENGTH = "L"
    TIME = "T"
    CURRENT = "I"
    TEMPERATURE = "Θ"
    VELOCITY = "V"
    ENERGY = "E"
    DIMENSIONLESS = "1"


@dataclass(frozen=True)
class PhysicalQuantity:
    """
    Physical quantity with a strictly typed formal Dimension Enum and explicit energy flow direction.
    Enforces the Concept Bottleneck: raw dictionaries or guessing via substring matching is forbidden.
    """
    magnitude: float
    dimension: Dimension                     # Forced type: Dimension Enum only.
    confidence: float = 1.0                  # Confidence from neural Dimension Head
    is_input: bool = False                   # Explicit energy direction (replaces 'in' string guessing)
    is_output: bool = False                  # Explicit energy direction (replaces 'out' string guessing)

    @property
    def value(self) -> float:
        """Backwards-compatible accessor for magnitude."""
        return self.magnitude

    @property
    def dimension_tag(self) -> str:
        """Backwards-compatible accessor for dimension name."""
        return self.dimension.name


# Backwards compatibility alias
DimensionalQuantity = PhysicalQuantity


@dataclass(frozen=True)
class MathHypothesis(Hypothesis):
    """
    Mathematical statement to be proven: lhs == rhs.
    Parsed via safe AST with zero eval/exec.
    Supports algebraic identities and interval bounds for non-linear functions.
    """
    lhs: str
    rhs: str
    tolerance: float = 1e-5
    var_bounds: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass(frozen=True)
class PhysicsHypothesis(Hypothesis):
    """
    Physical configuration evaluated on formal dimensional signatures,
    not arbitrary variable names.
    Only accepts PhysicalQuantity objects with Dimension Enum.
    """
    quantities: Union[List[PhysicalQuantity], Dict[str, PhysicalQuantity]] = field(default_factory=list)
    # Deprecated raw dictionary field (rejected by verifier if non-empty)
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialLogicHypothesis(Hypothesis):
    """
    Tier 1 Spatial Hypothesis: High-level topological and logic permissions.
    Evaluated by Clingo ASP on an abstracted room/zone graph to prevent state space explosion.
    """
    start_zone: str
    target_zone: str
    zone_sequence: List[str]
    allowed_transitions: Set[Tuple[str, str]]
    locked_doors: Set[str] = field(default_factory=set)
    agent_keys: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class KinematicTrajectoryHypothesis(Hypothesis):
    """
    Tier 2 Spatial Hypothesis: Continuous physics & trajectory collision.
    Evaluated by continuous geometric collision solver on line-segments/convex obstacles.
    """
    start_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    waypoints: List[Tuple[float, float]]
    obstacles: List[Tuple[float, float, float, float]]  # (min_x, min_y, max_x, max_y)
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)


# Backward-compatible alias for existing grid spatial tests
@dataclass(frozen=True)
class SpatialHypothesis(Hypothesis):
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]]
    walls: Set[Tuple[int, int]] = field(default_factory=set)
    hazards: Set[Tuple[int, int]] = field(default_factory=set)
    grid_size: int = 10


@dataclass(frozen=True)
class VerificationResult:
    """Structured proof certificate from the symbolic verifier."""
    is_valid: bool
    proof_log: str
    prover_name: str
    violation_details: Optional[str] = None
