"""Procedural Zero-Shot Logic Puzzle Generator for Text Sim Benchmarks."""

import random
from typing import Dict, Any, List

class LogicPuzzleGenerator:
    """Generates procedurally novel arithmetic and relational constraint puzzles."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_arithmetic_puzzle(self) -> Dict[str, Any]:
        """Generates out-of-range arithmetic word problem."""
        a = random.randint(10, 100)
        b = random.randint(5, 50)
        multiplier = random.choice([2, 3, 4, 5])
        
        # Problem: Base power is A MW, auxiliary capacitor adds B MW, then system scales by multiplier.
        expected = (a + b) * multiplier
        text = (
            f"Reactor core base power is {a} MW. The auxiliary capacitor adds {b} MW. "
            f"When overdrive mode scales the combined output by a factor of {multiplier}, "
            f"what is the total power output?"
        )
        z3_equation = f"( {a} + {b} ) * {multiplier} = {expected}"

        return {
            "type": "arithmetic",
            "prompt": text,
            "operands": [float(a), float(b), float(multiplier)],
            "ground_truth": float(expected),
            "z3_equation": z3_equation
        }

    def generate_conservation_puzzle(self) -> Dict[str, Any]:
        """Generates thermodynamic/energy conservation constraint problem."""
        e_in = random.randint(100, 500)
        loss = random.randint(10, 80)
        expected_out = e_in - loss

        text = (
            f"An arc reactor input delivers {e_in} MJ. Thermal radiation loss absorbs {loss} MJ. "
            f"Assuming conservation of energy, what is the maximum available usable energy output?"
        )

        return {
            "type": "physics_conservation",
            "prompt": text,
            "e_in": float(e_in),
            "loss": float(loss),
            "ground_truth": float(expected_out),
            "z3_properties": {"energy_in": float(e_in), "energy_out": float(expected_out)}
        }

    def generate_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        batch = []
        for i in range(count):
            if i % 2 == 0:
                batch.append(self.generate_arithmetic_puzzle())
            else:
                batch.append(self.generate_conservation_puzzle())
        return batch
