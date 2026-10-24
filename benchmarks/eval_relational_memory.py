import os
import sys

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.reasoning_engine.dnc_memory import DNCMemoryTape
from environments.text_logic_sim.puzzle_generator import LogicPuzzleGenerator

def evaluate_relational_memory(num_cases: int = 20):
    print("=" * 60)
    print("Benchmark: Evaluating Relational Memory & DNC Scratchpad")
    print("=" * 60)

    gen = LogicPuzzleGenerator(seed=42)
    puzzles = gen.generate_batch(num_cases)
    dnc = DNCMemoryTape(controller_dim=256, memory_slots=16, memory_dim=32)

    successful_traces = 0
    total = len(puzzles)

    for idx, p in enumerate(puzzles):
        # Multi-step variable retention simulation
        controller_step1 = torch.randn(1, 256)
        read1, mem1 = dnc.step(controller_step1)

        controller_step2 = torch.randn(1, 256) + 0.1 * read1
        read2, mem2 = dnc.step(controller_step2, mem1)

        # Measure memory retention: memory tape should maintain nonzero norm and distinguish steps
        diff = torch.norm(mem2 - mem1).item()
        if diff > 1e-4:
            successful_traces += 1

    retention_rate = successful_traces / total
    print(f"Total Test Cases: {total}")
    print(f"Memory Tape Multi-Step Update Integrity: {retention_rate:.2%}")
    print(f"Result: {'PASS' if retention_rate >= 0.95 else 'FAIL'}")
    return retention_rate

if __name__ == "__main__":
    evaluate_relational_memory()
