"""Neural Arithmetic Logic Units (NALU) with Numerical Stabilization & Semantic Typing."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any


class NeuralArithmeticLogicUnit(nn.Module):
    """
    Numerically stabilized NALU cell.
    Implements bounded log-space transforms to guarantee mathematical stability
    against NaN or floating-point overflow.
    """

    def __init__(self, in_features: int, out_features: int, epsilon: float = 1e-7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon

        # NAC parameters for exact [-1, 0, +1] linear combinations
        self.w_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.m_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.g_gate = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)
        nn.init.xavier_uniform_(self.g_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Constrain W to [-1, 1]
        w = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)

        # 1. Additive path
        a = torch.matmul(x, w.t())

        # 2. Multiplicative/Division path with strict numerical clamp
        log_x = torch.log(torch.abs(x) + self.epsilon)
        log_mult = torch.matmul(log_x, w.t())
        # Bounded between exp(-15) (~3e-7) and exp(15) (~3.2e6) to prevent NaN/Inf overflow
        log_mult_clamped = torch.clamp(log_mult, min=-15.0, max=15.0)
        m = torch.exp(log_mult_clamped)

        # 3. Gating
        g = torch.sigmoid(torch.matmul(x, self.g_gate.t()))

        out = g * a + (1.0 - g) * m
        return out


class NALUCluster(nn.Module):
    """
    Dedicated arithmetic cluster inside Jane's reasoning engine.
    Safely executes mathematical interpretations of SemanticTokens and projects
    exact mathematical facts directly into the 256D manifold.
    """

    def __init__(self, latent_dim: int = 256, math_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.math_dim = math_dim

        self.latent_to_num = nn.Linear(latent_dim, math_dim)
        self.layers = nn.ModuleList([
            NeuralArithmeticLogicUnit(math_dim, math_dim) for _ in range(num_layers)
        ])
        self.num_to_latent = nn.Sequential(
            nn.Linear(math_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def execute_semantic_token_math(self, semantic_tokens: List[Any]) -> List[float]:
        """
        Calculates mathematical values from SemanticTokens inside the math unit.
        Separation of concerns: tokenizer only tags types; NALU calculates the math.
        """
        values = []
        for tok in semantic_tokens:
            if getattr(tok, "raw_value", None) is not None:
                if tok.token_type == "PERCENTAGE":
                    # Math unit executes the percentage arithmetic
                    values.append(tok.raw_value / 100.0)
                else:
                    values.append(tok.raw_value)
        return values

    def forward(
        self,
        latent_state: torch.Tensor,
        explicit_numbers: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_state = self.latent_to_num(latent_state)

        if explicit_numbers is not None and explicit_numbers.numel() > 0:
            k = min(explicit_numbers.size(-1), self.math_dim)
            num_state = torch.cat([explicit_numbers[:, :k], num_state[:, k:]], dim=-1)

        val = num_state
        for layer in self.layers:
            val = layer(val)

        math_anchor = self.num_to_latent(val)
        return math_anchor, val
