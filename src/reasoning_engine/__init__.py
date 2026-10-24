"""Reasoning Engine: Native math, differentiable scratchpad, energy constraints, and manifold."""

from .nalu_arithmetic import NALUCluster, NeuralArithmeticLogicUnit
from .dnc_memory import DNCMemoryTape
from .energy_constraints import EnergyConstraintLandscape, PhysicsBarrier
from .manifold import RelationalManifold
from .hebbian_manifold import HebbianConceptManifold
from .tom_manifold import TheoryOfMindManifold

__all__ = [
    "NALUCluster",
    "NeuralArithmeticLogicUnit",
    "DNCMemoryTape",
    "EnergyConstraintLandscape",
    "PhysicsBarrier",
    "RelationalManifold",
    "HebbianConceptManifold",
    "TheoryOfMindManifold",
]


