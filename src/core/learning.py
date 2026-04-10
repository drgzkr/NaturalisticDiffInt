"""
NMPH (Nonmonotonic Plasticity Hypothesis) learning rules.

Two implementations:
1. BCMLearningRule  — BCM (Bienenstock-Cooper-Munro) rule; analytically cleanest.
2. NMPHLearningRule — Piecewise U-shaped rule closer to Ritvo et al.'s coactivity function.

Both operate on coactivity (product of pre- and post-synaptic activations) and produce
weight changes that weaken connections for moderate coactivity (differentiation zone)
and strengthen connections for high coactivity (integration zone).

References:
    Bienenstock, Cooper & Munro (1982) Journal of Neuroscience
    Norman et al. (2006) Psychological Review (NMPH)
    Ritvo et al. (2024) eLife — Methods section
"""

import torch
import torch.nn as nn


class BCMLearningRule(nn.Module):
    """
    BCM (Bienenstock-Cooper-Munro) learning rule with sliding modification threshold.

    Weight update:
        δw(x→y) = η * y * (y − θ_M) * x

    where:
        x   = pre-synaptic activation
        y   = post-synaptic activation
        θ_M = modification threshold (EMA of y²)

    Threshold update:
        θ_M ← τ * θ_M + (1 − τ) * y²

    Interpretation:
        y < θ_M  → δw < 0 (weakening; differentiation zone)
        y > θ_M  → δw > 0 (strengthening; integration zone)
        y ≈ 0    → δw ≈ 0 (inactive competitor; no change)

    Parameters
    ----------
    lr : float
        Learning rate η.
    tau : float
        Momentum for sliding threshold update. Higher τ → slower threshold adaptation.
    theta_init : float
        Initial value of the modification threshold θ_M.
    max_weight : float
        Weight clamp upper bound.
    min_weight : float
        Weight clamp lower bound.
    """

    def __init__(
        self,
        lr: float = 0.01,
        tau: float = 0.9,
        theta_init: float = 0.5,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ):
        super().__init__()
        self.lr = lr
        self.tau = tau
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.register_buffer("theta_M", torch.tensor(theta_init))

    def update_threshold(self, post_act: torch.Tensor) -> None:
        """Update the sliding modification threshold based on post-synaptic activity."""
        mean_sq = (post_act ** 2).mean()
        self.theta_M = self.tau * self.theta_M + (1 - self.tau) * mean_sq

    def delta_w(
        self, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weight change matrix.

        Parameters
        ----------
        pre_act  : shape (n_pre,)
        post_act : shape (n_post,)

        Returns
        -------
        dw : shape (n_post, n_pre)
            Weight change matrix; rows = post-synaptic, columns = pre-synaptic.
        """
        # BCM: δw = η * y * (y − θ_M) * x
        # Outer product: (n_post, 1) × (1, n_pre) → (n_post, n_pre)
        bcm_factor = post_act * (post_act - self.theta_M)  # (n_post,)
        dw = self.lr * torch.outer(bcm_factor, pre_act)
        return dw

    def apply(
        self, weights: torch.Tensor, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply BCM update to a weight matrix.

        Parameters
        ----------
        weights : shape (n_post, n_pre)
        pre_act : shape (n_pre,)
        post_act : shape (n_post,)

        Returns
        -------
        new_weights : shape (n_post, n_pre)
        """
        dw = self.delta_w(pre_act, post_act)
        new_weights = weights + dw
        new_weights = torch.clamp(new_weights, self.min_weight, self.max_weight)
        self.update_threshold(post_act)
        return new_weights

    def forward(
        self, weights: torch.Tensor, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        return self.apply(weights, pre_act, post_act)


class NMPHLearningRule(nn.Module):
    """
    Piecewise U-shaped NMPH learning rule based on coactivity.

    Coactivity: c(x, y) = act_x * act_y

    Weight change:
        c ∈ [0, θ_low)           → δw = 0         (inactive zone)
        c ∈ [θ_low, θ_cross)     → δw = -lr_weak * (c - θ_low)   (weakening zone)
        c ∈ [θ_cross, 1]         → δw = +lr_strong * (c - θ_cross) (strengthening zone)

    This piecewise linear shape is a close approximation to the coactivity function
    described (but not fully formalised) in Ritvo et al. (2024).

    Parameters
    ----------
    theta_low : float
        Minimum coactivity for any weight change (below = inactive competitor).
    theta_cross : float
        Crossover coactivity: below → weakening, above → strengthening.
    lr_weak : float
        Learning rate in the weakening (differentiation) zone.
    lr_strong : float
        Learning rate in the strengthening (integration) zone.
    max_weight, min_weight : float
        Weight bounds.
    """

    def __init__(
        self,
        theta_low: float = 0.05,
        theta_cross: float = 0.3,
        lr_weak: float = 0.05,
        lr_strong: float = 0.02,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ):
        super().__init__()
        self.theta_low = theta_low
        self.theta_cross = theta_cross
        self.lr_weak = lr_weak
        self.lr_strong = lr_strong
        self.max_weight = max_weight
        self.min_weight = min_weight

    def _u_shape(self, c: torch.Tensor) -> torch.Tensor:
        """Apply piecewise U-shaped function to coactivity scalar or tensor."""
        dw = torch.zeros_like(c)
        # Weakening zone
        weak_mask = (c >= self.theta_low) & (c < self.theta_cross)
        dw[weak_mask] = -self.lr_weak * (c[weak_mask] - self.theta_low)
        # Strengthening zone
        strong_mask = c >= self.theta_cross
        dw[strong_mask] = self.lr_strong * (c[strong_mask] - self.theta_cross)
        return dw

    def delta_w(
        self, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weight change matrix from pre/post activities.

        Returns
        -------
        dw : shape (n_post, n_pre)
        """
        # Coactivity matrix: c_ij = post_i * pre_j
        coactivity = torch.outer(post_act, pre_act)  # (n_post, n_pre)
        dw = self._u_shape(coactivity)
        return dw

    def apply(
        self, weights: torch.Tensor, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        dw = self.delta_w(pre_act, post_act)
        new_weights = torch.clamp(weights + dw, self.min_weight, self.max_weight)
        return new_weights

    def forward(
        self, weights: torch.Tensor, pre_act: torch.Tensor, post_act: torch.Tensor
    ) -> torch.Tensor:
        return self.apply(weights, pre_act, post_act)
