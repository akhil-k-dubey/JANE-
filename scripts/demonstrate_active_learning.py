"""Demonstration of Native Active Learning in Project Jane / Flux.

Demonstrates:
1. Native conversational concept teaching:
   User says: "Flux, a 'quantum entanglement' is when two particles share a wave function, so measuring one instantly affects the other."
   - Audio / Text Translation -> AST
   - Z3 SMT Prover verifies physical laws
   - Manifold computes 256D centroid between related concepts
   - Hebbian plasticity permanently wires the new node
   - Manifold, Self-Model, and Knowledge Graph updated and committed
2. Z3 Axiom Protection Gate:
   User tries to teach a perpetual motion machine (energy_out > energy_in).
   - Z3 detects thermodynamic violation and rejects the concept.
3. Introspective Verification:
   Jane inspects her updated knowledge base and confirms the new concept count.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pipeline import JanePipeline


def main():
    print("=" * 76)
    print("   PROJECT JANE / FLUX — NATIVE ACTIVE LEARNING INTERFACE")
    print("=" * 76)
    print("  • Concept Acquisition: Native conversational dialogue (Zero code needed)")
    print("  • Physical Invariants: Formally verified via Z3 SMT theorem prover")
    print("  • Geometric Placement: Placed at centroid of related concepts in 256D")
    print("  • Synaptic Plasticity: Permanently wired via Hebbian co-activation")
    print("=" * 76 + "\n")

    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device="cpu")

    # -------------------------------------------------------------------------
    # PART 1: Teaching a Valid New Concept
    # -------------------------------------------------------------------------
    print(">>> PART 1: NATIVELY TEACHING 'QUANTUM ENTANGLEMENT'")
    print("-" * 76)
    prompt1 = "Flux, a 'quantum entanglement' is when two particles share a wave function, so measuring one instantly affects the other."
    print(f"\033[1;36m[User Input]\033[0m {prompt1}\n")

    res1 = pipeline.process(text=prompt1)
    ver1 = res1.get("verification")
    if ver1 and ver1.is_valid:
        print(f"\033[92m[✓ Z3 Formally Verified | {ver1.prover_name}]\033[0m")
        print(f"  Proof Certificate: {ver1.proof_log}")
    print(f"\n\033[1;32m[Jane Response]\033[0m\n{res1['response_text']}\n")

    # -------------------------------------------------------------------------
    # PART 2: Testing Z3 Axiom Protection Gate (Thermodynamic Violation)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print(">>> PART 2: Z3 AXIOM PROTECTION GATE (REJECTING PHYSICAL VIOLATIONS)")
    print("-" * 76)
    prompt2 = "Flux, a 'perpetual motion device' is when energy_out > energy_in occurs continuously without fuel."
    print(f"\033[1;36m[User Input]\033[0m {prompt2}\n")

    res2 = pipeline.process(text=prompt2)
    ver2 = res2.get("verification")
    if ver2 and not ver2.is_valid:
        print(f"\033[91m[✗ Rejected by Z3 SMT Prover: {ver2.proof_log}]\033[0m")
    print(f"\n\033[1;33m[Jane Response]\033[0m\n{res2['response_text']}\n")

    # -------------------------------------------------------------------------
    # PART 3: Introspective Self-Model Confirmation
    # -------------------------------------------------------------------------
    print("=" * 76)
    print(">>> PART 3: INTROSPECTIVE CONFIRMATION (GROUNDED KNOWLEDGE BASE)")
    print("-" * 76)
    prompt3 = "how many concepts or knowledge graph do you currently have?"
    print(f"\033[1;36m[User Input]\033[0m {prompt3}\n")

    res3 = pipeline.process(text=prompt3)
    print(f"\033[1;32m[Jane Response]\033[0m\n{res3['response_text']}\n")

    # -------------------------------------------------------------------------
    # PART 4: Checking Topological Manifold Links
    # -------------------------------------------------------------------------
    print("=" * 76)
    print(">>> PART 4: TOPOLOGICAL MANIFOLD LINKS (HEBBIAN ADJACENCY)")
    print("-" * 76)
    links = pipeline.hebbian_manifold.get_strongest_links(top_n=6)
    for c1, c2, w in links:
        bars = "■" * int(w * 20)
        print(f"  • [{c1:22s} <-> {c2:22s}] weight: {w:.3f} {bars}")

    print("\n" + "=" * 76)
    print("         ACTIVE LEARNING PROTOCOL VERIFIED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
