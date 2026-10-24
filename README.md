# Project Jane

> **A Neuro-Symbolic Cognitive Engine & Continuous Latent Manifold Runtime**  
> *Evolution of the FLUX Cognitive Architecture*

---

## 1. System Overview

Standard Large Language Models (LLMs) model intelligence as discrete autoregressive token classification over a fixed vocabulary:

$$P(x_t \mid x_{<t}) = \text{Softmax}(W_h \cdot h_t)$$

While effective for broad surface linguistic fluency, this formulation has structural failure modes in scientific, physical, and real-time operational contexts:

1. **Arithmetic Approximation**: Numbers are parsed as arbitrary sub-word tokens. Arithmetic operations are statistically interpolated from text rather than calculated algorithmically.
2. **Physical Hallucinations**: Standard models optimize for likelihood under a training distribution. When prompted with scenarios involving conservation laws, kinetics, or spatial boundaries, they readily assert physically impossible states (negative mass, unbounded energy output, spatial phase-through).
3. **Context Window Forgetting**: Conversations rely on finite KV caches. When an operational project extends over days or weeks, early goals and foundational intent are evicted from context, causing the agent to lose its original objective.
4. **Zero Latent Deliberation**: Autoregressive decoding commits to output tokens in a single forward pass ($\mathcal{O}(1)$ computation depth per token) without verifying intermediate mathematical state.

**Project Jane** addresses these vulnerabilities by decoupling cognition, state retention, and arithmetic from language generation:

```
Human Utterance / Perception
          │
          ▼
┌────────────────────────────────────────┐
│ PretrainedLinguisticEar (MiniLM-L6-v2) │ ──► Projects text to 256D normalized vector
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ Continuous Concept Manifold (256D)     │ ──► Online Hebbian plasticity + EWC consolidation
├────────────────────────────────────────┤
│ Neural Arithmetic Logic Units (NALU)   │ ──► Deterministic +, -, *, / outside training range
├────────────────────────────────────────┤
│ Differentiable Neural Computer (DNC)   │ ──► 64 x 256 scratchpad tape for multi-step state
├────────────────────────────────────────┤
│ Energy-Based Model (EBM Physics)       │ ──► Infinite energy barrier penalties on invalid states
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ Deterministic Provers (Z3 SMT, Clingo) │ ──► First-Order Logic & ASP proof gatekeeper
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ PretrainedLinguisticMouth (Qwen2.5-3B) │ ──► In-process VRAM decoding & streaming
└────────────────────────────────────────┘
```

The system thinks in mathematics, differential state dynamics, and logic rules. The LLM is used strictly as a **linguistic transducer** (voice synthesis from proven latent states), never as the authoritative reasoning engine.

---

## 2. Core Architecture & Mathematical Mechanics

### 2.1 The Continuous Relational Concept Manifold ($\mathcal{M}^{256}$)
Concepts reside in a continuous 256-dimensional unit hypersphere:

$$z \in \mathbb{R}^{256}, \quad \|z\|_2 = 1$$

- **Topology & Online Plasticity** (`src/reasoning_engine/hebbian_manifold.py`): Concepts are interconnected via a learnable adjacency matrix $A \in \mathbb{R}^{N \times N}$. When related concepts activate together, edge weights update via continuous Hebbian dynamics:
  $$\Delta A_{ij} = \eta \cdot (z_i \cdot z_j - \gamma A_{ij})$$
- **Anti-Catastrophic Forgetting (EWC)** (`src/reasoning_engine/ewc_consolidation.py`): To prevent new online associations from overwriting foundational physics and math embeddings, updates are constrained by the diagonal of the empirical Fisher Information Matrix $F$:
  $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \frac{\lambda_{\text{ewc}}}{2} \sum_{i} F_i (\theta_i - \theta_i^*)^2, \quad \lambda_{\text{ewc}} = 5000.0$$

### 2.2 Neural Arithmetic Logic Units (NALU)
Standard linear layers guess numbers along training curves. The NALU (`src/reasoning_engine/nalu_arithmetic.py`) enforces strict arithmetic operations:

1. **Neural Accumulator (NAC)**: Constrains linear transformation weights to $\{-1, 0, +1\}$ via hyperbolic tangent and sigmoid gating:
   $$W_{\text{NAC}} = \tanh(V) \odot \sigma(M)$$
   $$a = W_{\text{NAC}} \cdot x \quad (\text{Exact Addition / Subtraction})$$
2. **Logarithmic Multiplication/Division**: Operates in log space to calculate products and quotients:
   $$m = \exp\left( W_{\text{NAC}} \cdot \log(|x| + \epsilon) \right)$$
3. **Operation Gate**:
   $$g = \sigma(G \cdot x), \quad y = g \odot a + (1 - g) \odot m$$

Values extracted from text (e.g., *"increase output to 40%"*) are parsed into NALU clusters. Calculated outputs anchor into the manifold as deterministic scalar coordinates.

### 2.3 Differentiable Neural Computer (DNC Memory Tape)
Multi-step deductions cannot be executed safely in a single forward pass. Jane incorporates a differentiable memory tape (`src/reasoning_engine/dnc_memory.py`):

- **Memory Matrix**: $M_t \in \mathbb{R}^{S \times W}$ ($S = 64$ slots, $W = 256$ dimensions).
- **Addressing Mechanism**: Cosine content-addressing combined with temporal link matrices tracks intermediate variables across reasoning steps.
- **Write / Erase**:
  $$M_t(i) = M_{t-1}(i) \odot (1 - w_t^w(i) e_t) + w_t^w(i) a_t$$
Jane writes intermediate mathematical values to the tape, steps through symbolic rules, reads back accumulated facts, and verifies state consistency before surface realization.

### 2.4 Energy-Based Physics Invariants (EBM)
Physical boundary conditions are formulated as an energy landscape (`src/reasoning_engine/energy_constraints.py`):

$$E_{\text{total}}(z) = E_{\text{learned}}(z) + E_{\text{physics}}(z)$$

Violations of physical invariants (e.g., negative mass, energy output exceeding input, spatial interpenetration) incur quadratic barrier penalties:

$$E_{\text{physics}}(z) = \sum_{k} \max\left(0, h_k(z)\right)^2 \cdot K_{\text{penalty}}, \quad K_{\text{penalty}} = 10^6$$

During latent space settling, states descend the energy gradient $\nabla_z E_{\text{total}}(z)$. Non-physical configurations act as potential barriers, forcing the trajectory exclusively into valid physical valleys.

### 2.5 Deterministic Symbolic Gatekeepers (Z3 & Clingo)
Statistical outputs are formally verified before real-world actuation:

- **Z3 SMT Solver** (`src/symbolic_verifier/z3_interface.py`, `src/symbolic_verifier/z3_code_verifier.py`): Translates numeric assertions and kinematic parameters into first-order Satisfiability Modulo Theories formulas. If a hypothesis asserts $E_{\text{out}} > E_{\text{in}}$ or an acceleration profile exceeds structural load tolerances, Z3 returns `UNSAT`/violation and halts execution.
- **Clingo ASP Solver** (`src/symbolic_verifier/clingo_interface.py`): Answer Set Programming handles spatial discrete navigation, grid topology, and conflict-free routing without probabilistic guesswork.
- **Lark Python AST & Safe Compiler** (`src/symbolic_verifier/lark_python_parser.py`, `src/symbolic_verifier/sandboxed_compiler.py`): Validates code syntax, strips unsafe builtins (`os.system`, `eval`, arbitrary socket calls), and verifies invariants prior to sandboxed compilation.

### 2.6 The Genesis Node & Semantic Resurfacer
To eliminate the context window limit problem without unbounded token inflation, Jane implements a latent goal anchor system (`src/reasoning_engine/genesis_node.py`, `src/reasoning_engine/semantic_resurfacer.py`):

1. **Genesis Node Creation**: When a project is initiated (e.g., *"Jane, we are starting a project to redesign the reactor cooling system"*), the utterance is projected into a 256D vector and checked against a normalized centroid of project-initiation prototypes. If similarity exceeds $0.45$, a permanent `GenesisAnchor` is created and stored in `data/genesis_nodes.json`.
2. **Semantic Resurfacer**: A background daemon monitors the center-of-mass of Jane's current conversational state against all registered Genesis Anchors.
3. **Drift Detection**: If current actions diverge from the project's original goal (cosine similarity drops below $0.35$ after the grace period), Jane triggers a targeted proactive alert:
   > *"Sir, I notice we are bypassing the thermal coupling. Our original intent for this project was to increase thermodynamic efficiency. Is this bypass intentional?"*

This operates entirely in 256D geometric space, requiring zero regex patterns or hardcoded conditionals.

### 2.7 In-Process VRAM Verbalizer (Qwen2.5-3B-Instruct)
Linguistic verbalization runs locally via `llama-cpp-python` (`src/core/llm_backend.py`):

- **Direct VRAM Pinning**: All 36 layers are pinned to the GPU (`n_gpu_layers: -1`), consuming ~2.1 GB of VRAM.
- **Prompt KV Caching**: Jane's system prompt (Persona Matrix, Self-Model telemetry, Z3 state) is evaluated once and preserved in VRAM.
- **Streaming**: Tokens stream directly to the terminal with Time-To-First-Token (TTFT) $< 80\text{ ms}$.

---

## 3. Repository Structure

```
JANE/
├── configs/
│   ├── llm.yaml                     # Qwen2.5-3B GGUF context & VRAM parameters
│   ├── reasoning_engine.yaml        # NALU, DNC, Manifold, and Energy parameters
│   ├── transformer_config.yaml      # Dense embedding configurations
│   └── solver_rules/                # Formal physical and spatial invariant specifications
│       └── physics_rules.yaml
│
├── src/
│   ├── core/                        # Central orchestration & system runtime
│   │   ├── pipeline.py              # Main JanePipeline orchestrator
│   │   ├── interactive_chat.py      # Unified CLI REPL with ANSI telemetry
│   │   ├── llm_backend.py           # In-process llama-cpp-python singleton
│   │   ├── intent_router.py         # Continuous intent classifier on hypersphere
│   │   ├── anticipatory_engine.py   # State trajectory forecasting daemon
│   │   ├── action_dispatcher.py     # Sandboxed action executor & tool bridge
│   │   ├── curiosity_daemon.py      # Autonomous graph entropy reduction
│   │   ├── codebase_introspector.py # AST inspection of Jane's own source
│   │   ├── live_hot_reloader.py     # Live code patcher with atomic rollback
│   │   └── tools.py                 # File I/O, search, and system probes
│   │
│   ├── reasoning_engine/            # Mathematical & cognitive modules
│   │   ├── manifold.py              # 256D continuous relational concept space
│   │   ├── hebbian_manifold.py      # Plastic concept graph with continuous update rules
│   │   ├── nalu_arithmetic.py       # Neural Arithmetic Logic Units
│   │   ├── dnc_memory.py            # Differentiable Neural Computer tape
│   │   ├── energy_constraints.py    # Energy-Based Model physics barriers
│   │   ├── genesis_node.py          # Latent semantic project anchors
│   │   ├── semantic_resurfacer.py   # Latent goal divergence monitor
│   │   ├── ewc_consolidation.py     # Elastic Weight Consolidation
│   │   ├── self_model.py            # Self-state, hardware telemetry, capability limits
│   │   └── tom_manifold.py          # Theory of Mind user belief modeling
│   │
│   ├── symbolic_verifier/           # Deterministic proof gatekeepers
│   │   ├── z3_interface.py          # First-Order SMT solver interface
│   │   ├── z3_code_verifier.py      # SMT verification of generated Python logic
│   │   ├── clingo_interface.py      # Answer Set Programming for spatial topologies
│   │   ├── verifier_orchestrator.py # Gatekeeper verifying neural hypotheses
│   │   ├── hallucination_critic.py  # EBM + SMT rejection of ungrounded assertions
│   │   ├── sandboxed_compiler.py    # Restricted execution environment
│   │   └── lark_python_parser.py    # Grammar AST validator
│   │
│   ├── perception/                  # Multimodal sensor projections
│   │   ├── audio_encoder.py         # Pitch, cadence, and acoustic feature extractor
│   │   ├── continuous_visual_eye.py # Visual stream processing into manifold
│   │   └── vision_encoder.py        # Spatial convolutional feature maps
│   │
│   └── translator/                  # Linguistic interfaces
│       ├── pretrained_translator.py # Ear (MiniLM -> 256D) and Mouth (Qwen in VRAM)
│       ├── encoder.py               # Text-to-vector latent projection
│       └── decoder.py               # Vector-to-text decoder
│
├── environments/                    # Simulation & benchmark testbeds
│   ├── grid_world/                  # 10x10 spatial kinematics test environment
│   └── text_logic_sim/              # Procedural constraint reasoning puzzles
│
├── data/
│   ├── knowledge_graph/             # Seed concept triples & physics principles
│   └── models/                      # Quantized LLM weights (downloaded on setup)
│
├── benchmarks/                      # Quantitative evaluation suites
│   ├── eval_relational_memory.py    # DNC multi-step memory benchmarks
│   ├── eval_grid_world.py           # Spatial navigation without collisions
│   └── eval_hallucination_rate.py   # SMT proof rejection verification
│
├── tests/                           # Unit and integration test suite
├── training/                        # Training scripts for manifold stages
├── scripts/
│   ├── download_qwen.py             # Downloads Qwen2.5-3B Q4_K_M GGUF
│   └── setup_llm.sh                 # Builds llama-cpp-python with CUDA/Vulkan
├── main.py                          # Unified entrypoint
├── requirements.txt                 # Exact pinned dependencies
└── pyproject.toml                   # Build system metadata
```

---

## 4. Hardware Budget & Technical Stack

Tested on an **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** running Ubuntu Linux:

| Component | Resident Memory | Compute Target |
|:---|:---:|:---|
| **Qwen2.5-3B-Instruct (Q4_K_M)** | ~2,100 MiB | GPU (VRAM-pinned via `llama-cpp-python`) |
| **System Prompt KV Cache** | ~320 MiB | GPU VRAM |
| **MiniLM-L6-v2 Semantic Encoder** | ~90 MiB | GPU / CPU |
| **256D Manifold + Hebbian Layer** | ~15 MiB | PyTorch GPU Tensor |
| **DNC Memory Tape ($64 \times 256$)** | ~2 MiB | PyTorch GPU Tensor |
| **Z3 SMT & Clingo Provers** | ~40 MiB | Multi-threaded CPU C++ core |
| **Total VRAM Consumption** | **~2,525 MiB** | Leaves **~3.2 GB headroom** for OS, vision & audio |

### Primary Dependencies
- **Tensor Operations & Neural Modules**: `torch>=2.0.0`
- **Formal Verification**: `z3-solver>=4.12.0`, `clingo>=5.6.0`
- **Grammar & AST**: `lark>=1.1.0`, `sympy>=1.12`
- **In-Process Inference Engine**: `llama-cpp-python>=0.3.0`
- **Embeddings & NLP Tools**: `transformers>=4.38.0`, `huggingface-hub`

---

## 5. Pragmatic Evaluation

### 5.1 Concrete Advantages (Pros)
- **Deterministic Numerical Accuracy**: Mathematical expressions evaluated through NALU and Z3 are exact. They do not drift or hallucinate on out-of-distribution numbers.
- **Formal Invariant Guarantee**: Verified claims are checked against physical constraints ($E_{\text{out}} \le E_{\text{in}}$, non-negative masses, kinematic continuity). Physically impossible outputs are rejected by construction.
- **Zero-Token Long-Term Retention**: Project goals anchored as Genesis Nodes persist indefinitely in a JSON ledger and 256D geometric space, bypassing token budget limits and KV cache eviction.
- **Fully Local & Contained**: No external cloud API calls, zero telemetry, fully operational offline on a single consumer GPU with $\le 6\text{ GB}$ VRAM.
- **Auditable Provenance**: All inferences, state transitions, and proof decisions are recorded chronologically in an immutable JSONL episodic ledger.

### 5.2 Real-World Limitations & Failure Modes
- **SMT Solvability Limits**: First-Order SMT solving over non-linear real arithmetic is undecidable in the general case and NP-hard in practice. Extremely complex or over-constrained physical systems will trigger timeouts (default: 5.0 seconds), forcing fallback to safe conservative postures.
- **Cosine Centroid Sensitivity**: The Intent Router and Genesis Node detector rely on cosine distance against seed prototype centroids. Nuanced or ambiguous project statements that hover near the threshold ($0.45$) can fail to register as project anchors unless phrased with clear operational intent.
- **Memory Tape Size Constraints**: The DNC scratchpad is parameterized with 64 slots of 256 dimensions. It is designed for intermediate variable chaining across multi-step problems, not for long-form textual document retrieval.
- **Pipeline Latency**: Decoupled multi-stage execution (Ear projection $\to$ Manifold settling $\to$ SMT proof check $\to$ LLM verbalization) introduces a 100–300 ms pipeline overhead compared to unverified single-pass autoregressive streaming.
- **Plasticity Drift**: Continuous online Hebbian learning can cause edge weight saturation if exposed to repetitive or narrow conversational domains over extended periods, requiring periodic EWC consolidation passes.

---

## 6. Installation & Quickstart

### Prerequisites
- Linux (Ubuntu 22.04+ recommended; tested on Ubuntu 24.04 and 26.04)
- Python 3.10 to 3.14
- NVIDIA GPU with CUDA support (or Vulkan-compatible GPU)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/akhil-k-dubey/JANE-.git
cd JANE-

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install SMT & ASP Solvers
```bash
# Ubuntu / Debian
sudo apt-get install -y z3 clingo
```

### 3. Build `llama-cpp-python` with GPU Acceleration
If building on a machine with CUDA 12.x+:
```bash
# Set CUDA compiler and host compiler flags
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" pip install llama-cpp-python \
  --upgrade --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```

*(Alternatively, use `bash scripts/setup_llm.sh` for an automated installation)*.

### 4. Download Model Weights
Download the Qwen2.5-3B-Instruct quantized GGUF (~2.1 GB):
```bash
python scripts/download_qwen.py
```

### 5. Run Jane Interactively
```bash
python main.py
```

Inside the interactive REPL:
```
>>> Jane, we are starting a project to redesign the reactor cooling system.
[Genesis Node] Project intent anchored permanently: "Jane, we are starting a project to redesign the reactor cooling system."
[Jane]: Acknowledged. Reactor cooling redesign project intent is permanently anchored in the 256D manifold.

>>> /genesis
=== Permanent Genesis Anchors (Project Intents) ===
[1] Jane, we are starting a project to redesign the reactor cooling system.
    Created: 2026-10-04 17:26:54 | Activations: 0
    Similarity to current manifold state: 0.884
Resurfacer daemon: ACTIVE (Checking every 45.0s, divergence threshold: 0.35)

>>> Adjust reactor output to 40% and verify conservation laws
[Z3 Verifier]: Formal constraints checked. Proof: SAT (0 Hallucinations)
[Jane]: Power adjustment to 40% verified against thermal invariants. Trajectory cleared.
```

### 6. Run Test Suite
Verify all unit, integration, and prover tests:
```bash
pytest tests/ -v
```

---

## 7. License & Citation

Project Jane is licensed under the [Apache License 2.0](LICENSE).

```bibtex
@software{dubey2026jane,
  author = {Dubey, Akhil K.},
  title = {Project Jane: A Neuro-Symbolic Cognitive Engine and Continuous Latent Manifold Runtime},
  year = {2026},
  url = {https://github.com/akhil-k-dubey/JANE-}
}
```
