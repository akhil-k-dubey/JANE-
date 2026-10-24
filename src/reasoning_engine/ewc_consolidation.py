"""Elastic Weight Consolidation (EWC) for Project Jane.

Prevents catastrophic forgetting during continuous active learning and concept ingestion.
Computes Fisher Information Matrix diagonals F_i over foundational tasks to anchor
critical neural weights (NALU arithmetic, DNC tape controller, manifold projections).
Applies quadratic penalty: L_EWC = sum_i (lambda / 2) * F_i * (theta_i - theta_i^*)^2.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ElasticWeightConsolidator:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017):
    Anchors critical neural weights so continuous learning does not degrade foundational physics/math skills.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self._anchor_current_parameters()

    def _anchor_current_parameters(self):
        """Anchors reference parameters theta*."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.detach().clone()
                # Initialize uniform baseline Fisher diagonal
                self.fisher[name] = torch.ones_like(param) * 1e-3

    def estimate_fisher(self, sample_inputs: List[torch.Tensor], target_loss_fn):
        """
        Estimates the empirical Fisher Information Matrix diagonal:
        F_i = E [ (grad_theta_i log P(y|x))^2 ]
        """
        self.model.eval()
        fisher_accum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad
        }

        for inp in sample_inputs:
            self.model.zero_grad()
            loss = target_loss_fn(self.model(inp))
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data ** 2

        n_samples = max(1, len(sample_inputs))
        for name in fisher_accum:
            self.fisher[name] = (fisher_accum[name] / n_samples) + 1e-4

        self.model.train()

    def compute_ewc_penalty(self) -> torch.Tensor:
        """
        Computes quadratic EWC penalty loss:
        L_penalty = sum_i (lambda / 2) * F_i * (theta_i - theta_i^*)^2
        """
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.params and name in self.fisher:
                diff = param - self.params[name].to(param.device)
                f_diag = self.fisher[name].to(param.device)
                penalty += (self.ewc_lambda / 2.0) * torch.sum(f_diag * (diff ** 2))
        return penalty

    def consolidate_new_task(self):
        """Re-anchors weights to current values while accumulating Fisher information."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.detach().clone()
