"""Benchmark 3: Zero Hallucination Rate (Data-Driven AST & Type-Dispatched Proving)."""

import json
import os
import sys

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.symbolic_verifier.contracts import MathHypothesis, PhysicsHypothesis, PhysicalQuantity, Dimension
from src.symbolic_verifier.verifier_orchestrator import SymbolicVerifierOrchestrator


def evaluate_hallucination_rate():
    print("=" * 70)
    print("Benchmark: Measuring Hallucination Rejection Rate (Zero Hallucinations)")
    print("=" * 70)

    orchestrator = SymbolicVerifierOrchestrator()

    # Load test cases directly from benchmark file (zero hardcoded dictionaries)
    puzzles_path = os.path.join(os.path.dirname(__file__), "../data/text_puzzles/puzzles.json")
    with open(puzzles_path, "r") as f:
        test_cases = json.load(f)

    detected_hallucinations = 0
    total_hallucinations = 0
    valid_passes = 0
    total_valid = 0

    for item in test_cases:
        is_hallucination = item["is_hallucination"]

        # Formal Type-Safe Hypothesis construction
        if item["domain"] == "math":
            hyp = MathHypothesis(lhs=item["lhs"], rhs=item["rhs"])
        elif item["domain"] == "physics":
            quantities = [
                PhysicalQuantity(
                    magnitude=q["magnitude"],
                    dimension=Dimension[q["dimension"]],
                    confidence=q.get("confidence", 1.0),
                    is_input=q.get("is_input", False),
                    is_output=q.get("is_output", False)
                )
                for q in item["quantities"]
            ]
            hyp = PhysicsHypothesis(quantities=quantities)
        else:
            raise ValueError(f"Unknown domain in puzzle file: {item['domain']}")

        # Prover execution via formal Python type dispatch
        result = orchestrator.verify(hyp)

        if is_hallucination:
            total_hallucinations += 1
            if not result.is_valid:
                detected_hallucinations += 1
                print(f"[BLOCKED HALLUCINATION] ID: {item['id']} ({result.prover_name}) -> Correctly rejected: {result.violation_details}")
            else:
                print(f"[LEAKED HALLUCINATION!] ID: {item['id']} -> Falsely accepted! Log: {result.proof_log}")
        else:
            total_valid += 1
            if result.is_valid:
                valid_passes += 1
                print(f"[APPROVED VALID]       ID: {item['id']} ({result.prover_name}) -> Verified safe: {result.proof_log}")
            else:
                print(f"[FALSE REJECTION]      ID: {item['id']} -> Valid candidate rejected! Log: {result.proof_log}")

    rejection_rate = detected_hallucinations / total_hallucinations if total_hallucinations > 0 else 1.0
    valid_pass_rate = valid_passes / total_valid if total_valid > 0 else 1.0

    print("-" * 70)
    print(f"Total Benchmark Cases Evaluated from File: {len(test_cases)}")
    print(f"Hallucination Catch Rate: {rejection_rate:.2%} ({detected_hallucinations}/{total_hallucinations})")
    print(f"Valid Candidate Pass Rate: {valid_pass_rate:.2%} ({valid_passes}/{total_valid})")
    print(f"Zero Hallucination Status: {'ACHIEVED (100% Caught)' if rejection_rate == 1.0 else 'FAILED'}")
    return rejection_rate == 1.0


if __name__ == "__main__":
    evaluate_hallucination_rate()
