"""Project Jane - Phase 1: Neural Engine (The Empty Architecture).

Main CLI & interactive entry point demonstrating:
1. Semantic Tokenization (Typed Tokens without Mutation)
2. Native Math Routing via NALU
3. DNC Memory Tape Multi-Step Scratchpad
4. Relational Manifold (256D concept dots + 3D presentation)
5. Energy-Based Constraint Landscape (Continuous Physics Brick Walls)
6. Dynamic Latent Space System 2 Reasoning Loop (Equilibrium Convergence)
7. Deterministic Zero-Hallucination Verification (SymPy AST -> Z3 & Clingo ASP)
"""

import sys
import argparse
import os
from typing import Dict, Any
import torch

from src.core.pipeline import JanePipeline
from src.symbolic_verifier.contracts import (
    MathHypothesis,
    PhysicsHypothesis,
    SpatialHypothesis,
    PhysicalQuantity,
    Dimension
)


def safe_load_submodule(module: torch.nn.Module, state_dict: dict, name: str = "submodule") -> bool:
    """Safely loads weights into a PyTorch module with shape filtering and strict=False."""
    if not isinstance(state_dict, dict):
        return False
    model_state = module.state_dict()
    compatible = {}
    mismatched = []
    for k, v in state_dict.items():
        if k in model_state:
            if hasattr(v, "shape") and v.shape == model_state[k].shape:
                compatible[k] = v
            else:
                v_shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
                mismatched.append((k, v_shape, tuple(model_state[k].shape)))
        else:
            mismatched.append((k, getattr(v, "shape", type(v)), "missing in module"))

    incompatible = module.load_state_dict(compatible, strict=False)
    if mismatched:
        print(f"[Warning: Checkpoint '{name}' shape mismatches filtered: {len(mismatched)} keys skipped]")
    return len(compatible) > 0


def main():
    parser = argparse.ArgumentParser(description="Project Jane: Neuro-Symbolic Reasoning Engine")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal chat interface")
    parser.add_argument("--text", type=str, default=None, help="Input command")
    parser.add_argument("--domain", type=str, default="physics", choices=["physics", "math", "spatial", "dynamic"], help="Prover domain")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--checkpoint", type=str, default="logs/training_runs/phase3_world_manifold_rl.pt", help="Manifold checkpoint")
    args = parser.parse_args()

    # Default to interactive chat interface when called with no text or with --interactive
    if args.interactive or (args.text is None and "--domain" not in sys.argv):
        from src.core.interactive_chat import JaneTerminalChat
        chat = JaneTerminalChat(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        chat.run()
        return

    text_to_process = args.text or "Adjust reactor output to 40% and calibrate thruster"
    args.text = text_to_process

    print("=" * 70)
    print(f"PROJECT JANE — NEURAL REASONING ENGINE & WORLD MANIFOLD")
    print(f"Device: {args.device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 70)

    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=True, device=args.device)

    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=args.device)
            if isinstance(ckpt, dict):
                if "manifold" in ckpt:
                    safe_load_submodule(pipeline.manifold, ckpt["manifold"], "pipeline.manifold")
                if "ear_projection" in ckpt:
                    safe_load_submodule(pipeline.ear.projection, ckpt["ear_projection"], "pipeline.ear.projection")
                if "energy_model" in ckpt:
                    safe_load_submodule(pipeline.energy_landscape, ckpt["energy_model"], "pipeline.energy_landscape")
            print(f"[Loaded World Manifold Checkpoint: {args.checkpoint}]")
        except Exception as e:
            print(f"[Warning: Failed to load checkpoint from {args.checkpoint}: {e}]")

    if args.domain == "dynamic":
        from src.reasoning_engine.system2_planner import System2Planner
        from environments.spatial_graph.environment import SpatialEnvironmentGraph

        planner = System2Planner(latent_dim=256, device=args.device)
        if os.path.exists(args.checkpoint):
            try:
                ckpt = torch.load(args.checkpoint, map_location=args.device)
                if isinstance(ckpt, dict):
                    if "manifold" in ckpt:
                        safe_load_submodule(planner.manifold, ckpt["manifold"], "planner.manifold")
                    if "ear_projection" in ckpt:
                        safe_load_submodule(planner.ear.projection, ckpt["ear_projection"], "planner.ear.projection")
            except Exception as e:
                print(f"[Warning: Failed to load dynamic planner checkpoint from {args.checkpoint}: {e}]")

        env = SpatialEnvironmentGraph.create_facility_env()
        print(f"\n[1] Dynamic Multi-Step Instruction: '{args.text}'")
        print(f"[2] Running System 2 DNC Scratchpad Deliberation & Symbolic Boundary Checks...")

        audit = planner.execute_plan(env, args.text, allow_self_correction=True)

        print("\n--- SYSTEM 2 AUDIT TRAIL ---")
        for log in audit.inner_monologue_logs:
            print(f"  {log}")

        print("\n--- RESULTS ---")
        print(f"Plan Execution Status:  {'SUCCESS (100% Verified)' if audit.success else 'BLOCKED / REJECTED'}")
        print(f"DNC Steps Committed:    {audit.dnc_steps_executed} steps (Memory Norm: {audit.dnc_memory_norm:.4f})")
        print(f"Waypoints Traversed:    {len(audit.trajectory_waypoints)} continuous coordinates")
        print(f"Final Agent Position:   Room '{audit.final_environment_state['agent_node']}' at {audit.final_environment_state['agent_pos']}")
        print(f"Agent Inventory:        {audit.final_environment_state['inventory']}")
        print(f"Reactor Power State:    {audit.final_environment_state['reactor_power']}%")
        print("=" * 70)
        return

    # Formal Hypothesis construction (Formal Python Types, Zero String Checks)
    if args.domain == "physics":
        hypothesis = PhysicsHypothesis(quantities=[
            PhysicalQuantity(magnitude=25.0, dimension=Dimension.MASS, confidence=0.99),
            PhysicalQuantity(magnitude=100.0, dimension=Dimension.ENERGY, confidence=0.99, is_input=True),
            PhysicalQuantity(magnitude=80.0, dimension=Dimension.ENERGY, confidence=0.99, is_output=True)
        ])
    elif args.domain == "math":
        hypothesis = MathHypothesis(lhs="15 + 25", rhs="40")
    else:
        hypothesis = SpatialHypothesis(
            start=(0, 0), goal=(2, 2),
            path=[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
            walls={(0, 1)}, hazards=set()
        )

    print(f"\n[1] Input Text: '{args.text}'")
    print(f"[2] Running Dynamic System 2 Latent Reasoning (Equilibrium Settling)...")

    result = pipeline.process(
        text=args.text,
        hypothesis=hypothesis
    )

    print("\n--- RESULTS ---")
    print(f"Response Verbalization: {result['response_text']}")
    print(f"Converged at Step:      {result['converged_at_step']}")
    if result["verification"]:
        v = result["verification"]
        print(f"Formal Verification:    {'PASSED' if v.is_valid else 'REJECTED'}")
        print(f"Verifier Engine:        {v.prover_name}")
        print(f"Proof Log:              {v.proof_log}")
    print(f"Settled Latent Energy:  {result['final_energy']:.4f}")
    print(f"Energy Trajectory:      {[round(e, 3) for e in result['energy_trace']]}")
    print(f"3D Visual Dots Sample:  {result['dots_3d'][0][:2]} (Total Dots: {len(result['dots_3d'][0])})")
    print("=" * 70)


if __name__ == "__main__":
    main()
