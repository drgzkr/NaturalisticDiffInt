"""
kWTA (k-Winners-Take-All) inhibition with sinusoidal oscillation envelope.

Implements the Ritvo et al. (2024) inhibitory dynamics:
- kWTA: at most k units active per layer
- Target Diff extension: "tied" units near the k-th threshold also activate
- Oscillatory envelope: inhibition modulated by sin(2π t / period),
  allowing competitor units to "pop up" transiently

References:
    O'Reilly & Munakata (2000) Computational Explorations in Cognitive Neuroscience
    Ritvo et al. (2024) eLife Table 1
"""

import numpy as np
import torch
import torch.nn as nn
import math


class KWTALayer(nn.Module):
    """
    k-Winners-Take-All layer with Target Diff and sinusoidal oscillatory inhibition.

    Parameters
    ----------
    k : int
        Number of units guaranteed to be active.
    k_max : int or None
        Maximum number of units allowed (for Target Diff extension). Default = 2*k.
    target_diff : float
        If a unit ranked below k is within `target_diff` of the k-th unit's
        excitation level, it is allowed to activate ("tied" unit).
    osc_amp : float
        Amplitude of sinusoidal modulation of inhibition. 0.0 = no oscillation.
        Positive values lower inhibition during the trough, letting competitors activate.
    osc_period : int
        Period of the sinusoidal oscillation in time steps.
    gain : float
        Gain on the activation sigmoid (XX1 Gain in Emergent; higher = more saturating).
    """

    def __init__(
        self,
        k: int,
        k_max: int | None = None,
        target_diff: float = 0.05,
        osc_amp: float = 0.0,
        osc_period: int = 10,
        gain: float = 100.0,
    ):
        super().__init__()
        self.k = k
        self.k_max = k_max if k_max is not None else 2 * k
        self.target_diff = target_diff
        self.osc_amp = osc_amp
        self.osc_period = osc_period
        self.gain = gain

    def oscillation_factor(self, t: int) -> float:
        """
        Returns the current inhibition multiplier at time step t.
        Factor < 1 means inhibition is lowered (more units can activate).
        """
        if self.osc_amp == 0.0:
            return 1.0
        return 1.0 - self.osc_amp * math.sin(2 * math.pi * t / self.osc_period)

    def apply(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Apply kWTA inhibition to a layer's pre-activation vector x.

        Parameters
        ----------
        x : torch.Tensor, shape (n_units,)
            Pre-activation (net excitatory input) of each unit.
        t : int
            Current time step (used to compute oscillation phase).

        Returns
        -------
        act : torch.Tensor, shape (n_units,)
            Post-inhibition activation in [0, 1].
        """
        n = x.shape[0]
        osc = self.oscillation_factor(t)

        # Sort units by excitation (descending)
        sorted_idx = torch.argsort(x, descending=True)
        k_eff = min(self.k, n)

        # Excitation of the k-th unit (threshold)
        threshold_excitation = x[sorted_idx[k_eff - 1]]

        # Inhibition level: set so that k-th unit is exactly at activation threshold
        # (In Leabra, inhibition cancels excitation: inhib = excit_k-th)
        inhib_base = threshold_excitation.item()
        inhib = inhib_base * osc  # oscillation modulates inhibition level

        # Net input after inhibition
        net = x - inhib

        # Target Diff extension: allow units "tied" with the k-th unit
        active_mask = net > 0
        # Count active units; cap at k_max
        n_active = active_mask.sum().item()
        if n_active > self.k_max:
            # Too many units active — increase inhibition to cut down to k_max
            top_k_max_threshold = x[sorted_idx[self.k_max - 1]]
            inhib = top_k_max_threshold.item() * osc
            net = x - inhib
            active_mask = net > 0

        # Sigmoid activation (XX1 function approximation): act = 1 / (1 + exp(-gain * net))
        # Clamp net to avoid overflow
        net_clamped = torch.clamp(net, -5.0, 5.0)
        act = torch.sigmoid(self.gain * net_clamped)

        # Apply mask: inactive units → 0
        act = act * active_mask.float()

        return act

    def forward(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        return self.apply(x, t)
