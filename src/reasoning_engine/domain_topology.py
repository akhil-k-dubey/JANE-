"""Continuous Domain Topology & Density Balancing Engine for Project Jane.

Provides 256D geometric manifold domain classification, density monitoring,
and topological partitioning to prevent concept over-densification.
Eliminates brittle keyword matching and single-domain dumping.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from src.utils.logger import get_logger
from src.reasoning_engine.intent_manifold import ContinuousIntentManifold

logger = get_logger("domain_topology")


class DomainTopologyEngine:
    """
    256D Continuous Manifold Domain Classifier and Density Balancer.
    Classifies concepts, queries, and principles into neuro-symbolic domains
    and monitors topological density to avoid crowding single domains.
    """

    DOMAINS = [
        "Computer Science & Coding",
        "Physics & Energy",
        "Mathematics & Logic",
        "Cognitive & Epistemic",
        "Speech & Audio Modality"
    ]

    # Domain prototypes in 256D latent space
    PROTOTYPES = {
        "Computer Science & Coding": [
            "computer science coding programming language python software algorithms data structures source code modules",
            "turing machine computability church turing thesis compiler bytecode ast symbolic execution computation",
            "neural network architecture pytorch functions scripts modules code generation synthesis",
            "formal methods z3 clingo declarative programming computational complexity programming languages"
        ],
        "Physics & Energy": [
            "physics physical sciences classical mechanics thermodynamics energy power force mass acceleration velocity kinematic limits conservation laws",
            "quantum mechanics wave function superposition entanglement schrodinger wave particle duality heisenberg uncertainty",
            "relativity spacetime lorentz contraction mass energy equivalence c speed of light general relativity",
            "electrodynamics magnetic flux superconductivity meissner effect voltage current resistance stefan boltzmann ideal gas",
            "optics optical interference holographic wavefront laser reflection diffraction photons wave mechanics"
        ],
        "Mathematics & Logic": [
            "mathematics math mathematical arithmetic linear algebra calculus derivative integral differential equations limit functions variables",
            "mathematical logic boolean logic predicate calculus first order logic axioms theorems proofs symbolic truth",
            "hyperbolic non-euclidean geometry manifolds curvature topology metric space poincare minkowski",
            "probability statistics stochastic processes information entropy shannon bayesian distributions"
        ],
        "Cognitive & Epistemic": [
            "cognitive science psychology theory of mind ego alter perspective causal agency cognitive dissonance",
            "epistemic discovery popperian falsification hypothesis void exploration popper invariant",
            "self model introspective self awareness architecture capabilities memory ledger identity",
            "counterfactual reasoning deliberation planning cognitive monologue epistemic state"
        ],
        "Speech & Audio Modality": [
            "speech audio acoustics flux voice auditory perception vad silence gating spectrogram audio waveform",
            "tone modulation prosody phonetics listening pitch frequency timbre volume",
            "vocal anchors alter tone continuous ear passive eavesdropping microphone speaker"
        ]
    }

    _instance: Optional["DomainTopologyEngine"] = None

    @classmethod
    def get_instance(cls, latent_dim: int = 256) -> "DomainTopologyEngine":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(latent_dim=latent_dim)
        return cls._instance

    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim
        self.intent_manifold = ContinuousIntentManifold.get_instance(latent_dim=latent_dim)
        self.centroids: Dict[str, torch.Tensor] = {}
        self._build_domain_centroids()

    def _build_domain_centroids(self):
        """Constructs continuous 256D normalized centroids for each domain."""
        for dom, phrases in self.PROTOTYPES.items():
            vecs = [self.intent_manifold.encode(p) for p in phrases]
            stacked = torch.stack(vecs)
            centroid = torch.mean(stacked, dim=0)
            self.centroids[dom] = F.normalize(centroid, p=2, dim=-1)

    def classify_domain(self, text: str) -> Tuple[str, float]:
        """
        Classifies input text or concept into the closest 256D domain centroid.
        Returns: (best_matching_domain, cosine_similarity)
        """
        z = self.intent_manifold.encode(text)
        sims = {
            dom: F.cosine_similarity(z.unsqueeze(0), cent.unsqueeze(0)).item()
            for dom, cent in self.centroids.items()
        }
        best_domain, best_sim = max(sims.items(), key=lambda x: x[1])
        return best_domain, best_sim

    def monitor_density(
        self,
        domain_counts: Dict[str, int],
        density_ceiling: int = 50
    ) -> Dict[str, Any]:
        """
        Monitors domain concept densities across Jane's manifold.
        Detects imbalances and provides sub-clustering recommendations to prevent over-densification.
        """
        total = sum(domain_counts.values()) or 1
        report = {}
        for dom in self.DOMAINS:
            count = domain_counts.get(dom, 0)
            share = count / total
            is_dense = count >= density_ceiling
            report[dom] = {
                "count": count,
                "share": share,
                "is_dense": is_dense,
                "status": "BALANCED" if not is_dense else "SATURATED"
            }
        return report
