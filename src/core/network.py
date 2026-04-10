"""
NMPH Network — PyTorch reimplementation of the Ritvo et al. (2024) model.

Architecture (see paper Figure 2 and Methods):
    - category layer  : shared features (1 unit per category in the original; here n_category)
    - item layer      : unique item features (1 unit per item in original; here n_item)
    - hidden layer    : internal representation; site of competition and learning
    - output layer    : associated stimuli / predicted consequences

All projections are bidirectional and fully connected.
Hidden → Hidden recurrent connections also exist.

Learning uses NMPH (U-shaped coactivity rule) applied after each stimulus presentation.
Inhibitory dynamics use kWTA with sinusoidal oscillation.

This implementation departs from Emergent in:
- Using continuous (not rate-coded spiking) activation
- Using BCM or piecewise NMPH learning rather than Leabra's exact conductance model
- Using standard PyTorch tensors rather than Emergent's internal data structures

For the naturalistic extension, see MemoryArchive in memory_archive.py.

References:
    Ritvo et al. (2024) eLife 12:RP88608
    O'Reilly & Munakata (2000) Computational Explorations in Cognitive Neuroscience
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal

from .inhibition import KWTALayer
from .learning import BCMLearningRule, NMPHLearningRule


@dataclass
class NetworkConfig:
    """Configuration for NMPHNetwork."""
    n_category: int = 1
    n_item: int = 2          # number of distinct items (pairmates); usually 2
    n_hidden: int = 14
    n_output: int = 1
    # kWTA parameters (see Table 1 in Ritvo et al.)
    hidden_k: int = 6
    hidden_k_max: int = 10
    hidden_target_diff: float = 0.03
    hidden_osc_amp: float = 0.11
    hidden_osc_period: int = 10
    output_k: int = 6
    output_k_max: int = 15
    output_target_diff: float = 0.05
    output_osc_amp: float = 0.115
    # Learning parameters
    learning_rule: Literal["bcm", "nmph"] = "bcm"
    lr: float = 0.01
    tau_bcm: float = 0.9
    theta_low: float = 0.05    # NMPH only
    theta_cross: float = 0.3   # NMPH only
    lr_weak: float = 0.05      # NMPH only
    lr_strong: float = 0.02    # NMPH only
    # Weight initialisation
    wt_range_hidden: tuple = (0.45, 0.55)
    wt_range_output: tuple = (0.01, 0.03)
    prewire_strength: float = 0.99   # strength of pre-wired pairmate connections
    hidden_overlap: int = 2          # number of hidden units shared between pairmates A and B


class NMPHNetwork(nn.Module):
    """
    Minimal NMPH network (Ritvo et al. 2024, PyTorch reimplementation).

    Usage
    -----
    >>> cfg = NetworkConfig(n_hidden=14, hidden_osc_amp=0.11)
    >>> net = NMPHNetwork(cfg)
    >>> # encode pairmate A
    >>> h_A, o_A = net.present_stimulus(category_idx=0, item_idx=0, n_ticks=20)
    >>> # encode pairmate B (competitor A becomes active if similar)
    >>> h_B, o_B = net.present_stimulus(category_idx=0, item_idx=1, n_ticks=20)
    >>> # measure representational similarity
    >>> sim = net.pairmate_similarity()
    """

    def __init__(self, cfg: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.cfg = cfg
        self._build_layers()
        self._init_weights()
        self._prewire()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_layers(self):
        cfg = self.cfg
        # Weight matrices (n_post × n_pre); bidirectional = two separate matrices
        # Category → Hidden
        self.W_cat_hid = nn.Parameter(torch.zeros(cfg.n_hidden, cfg.n_category))
        self.W_hid_cat = nn.Parameter(torch.zeros(cfg.n_category, cfg.n_hidden))
        # Item → Hidden
        self.W_item_hid = nn.Parameter(torch.zeros(cfg.n_hidden, cfg.n_item))
        self.W_hid_item = nn.Parameter(torch.zeros(cfg.n_item, cfg.n_hidden))
        # Hidden → Output
        self.W_hid_out = nn.Parameter(torch.zeros(cfg.n_output, cfg.n_hidden))
        self.W_out_hid = nn.Parameter(torch.zeros(cfg.n_hidden, cfg.n_output))
        # Hidden → Hidden (recurrent)
        self.W_hid_hid = nn.Parameter(torch.zeros(cfg.n_hidden, cfg.n_hidden))

        # Inhibitory layers
        self.hidden_kwta = KWTALayer(
            k=cfg.hidden_k,
            k_max=cfg.hidden_k_max,
            target_diff=cfg.hidden_target_diff,
            osc_amp=cfg.hidden_osc_amp,
            osc_period=cfg.hidden_osc_period,
        )
        self.output_kwta = KWTALayer(
            k=cfg.output_k,
            k_max=cfg.output_k_max,
            target_diff=cfg.output_target_diff,
            osc_amp=cfg.output_osc_amp,
        )

        # Learning rule
        if cfg.learning_rule == "bcm":
            self.learning_rule = BCMLearningRule(lr=cfg.lr, tau=cfg.tau_bcm)
        else:
            self.learning_rule = NMPHLearningRule(
                theta_low=cfg.theta_low,
                theta_cross=cfg.theta_cross,
                lr_weak=cfg.lr_weak,
                lr_strong=cfg.lr_strong,
            )

        # Storage for last activations (for learning and analysis)
        self._last_hidden = torch.zeros(cfg.n_hidden)
        self._last_output = torch.zeros(cfg.n_output)
        self._last_item_act = torch.zeros(cfg.n_item)
        self._last_cat_act = torch.zeros(cfg.n_category)

        # Pairmate hidden representations (stored after each stimulus presentation)
        self._pairmate_hidden = {}

    def _init_weights(self):
        cfg = self.cfg
        lo_h, hi_h = cfg.wt_range_hidden
        lo_o, hi_o = cfg.wt_range_output

        for p in [self.W_cat_hid, self.W_hid_cat, self.W_item_hid, self.W_hid_item,
                  self.W_hid_hid]:
            nn.init.uniform_(p, lo_h, hi_h)

        for p in [self.W_hid_out, self.W_out_hid]:
            nn.init.uniform_(p, lo_o, hi_o)

        # Zero diagonal of recurrent weights (no self-connections)
        with torch.no_grad():
            self.W_hid_hid.data.fill_diagonal_(0.0)

    def _prewire(self):
        """
        Pre-wire strong connections for pairmates A and B in the hidden layer.
        Pairmate A: item unit 0 → hidden units 0..5 (strong)
        Pairmate B: item unit 1 → hidden units 8..13 (strong)
        Shared (overlap): hidden units 6 and 7 connect to both

        Adjust indices based on n_hidden and hidden_overlap.
        """
        cfg = self.cfg
        n_h = cfg.n_hidden
        overlap = cfg.hidden_overlap
        s = cfg.prewire_strength

        # Assign hidden units to A, B, and shared
        n_unique = (n_h - overlap) // 2
        A_units = list(range(n_unique))
        shared_units = list(range(n_unique, n_unique + overlap))
        B_units = list(range(n_unique + overlap, 2 * n_unique + overlap))

        with torch.no_grad():
            # Item 0 (pairmate A) → A-units and shared units
            for u in A_units + shared_units:
                if u < n_h:
                    self.W_item_hid.data[u, 0] = s
            # Item 1 (pairmate B) → B-units and shared units
            for u in B_units + shared_units:
                if u < n_h:
                    self.W_item_hid.data[u, 1] = s
            # Strong hidden→hidden within each pairmate group
            for u in A_units:
                for v in A_units:
                    if u != v and v < n_h:
                        self.W_hid_hid.data[u, v] = s * 0.5
            for u in B_units:
                for v in B_units:
                    if u != v and v < n_h and u < n_h:
                        self.W_hid_hid.data[u, v] = s * 0.5

        self._A_units = A_units
        self._B_units = B_units
        self._shared_units = shared_units

    # ------------------------------------------------------------------
    # Forward pass (one tick)
    # ------------------------------------------------------------------

    def _compute_excitation(
        self,
        cat_act: torch.Tensor,
        item_act: torch.Tensor,
        hid_act: torch.Tensor,
        out_act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute net excitation to hidden and output layers."""
        exc_hid = (
            self.W_cat_hid @ cat_act
            + self.W_item_hid @ item_act
            + self.W_hid_hid @ hid_act
            + self.W_out_hid @ out_act
        )
        exc_out = self.W_hid_out @ hid_act
        return exc_hid, exc_out

    def forward_tick(
        self,
        cat_act: torch.Tensor,
        item_act: torch.Tensor,
        hid_act: torch.Tensor,
        out_act: torch.Tensor,
        t: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One settling tick: compute excitation, apply kWTA.
        Returns new (hid_act, out_act).
        """
        exc_hid, exc_out = self._compute_excitation(cat_act, item_act, hid_act, out_act)
        new_hid = self.hidden_kwta(exc_hid, t)
        new_out = self.output_kwta(exc_out, t)
        return new_hid, new_out

    # ------------------------------------------------------------------
    # Stimulus presentation (settling loop)
    # ------------------------------------------------------------------

    def present_stimulus(
        self,
        category_idx: int,
        item_idx: int,
        n_ticks: int = 20,
        store_label: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Present one stimulus and settle the network for n_ticks.
        Oscillatory inhibition runs over the ticks.

        Parameters
        ----------
        category_idx : int
            Which category unit to clamp active (0-indexed).
        item_idx : int
            Which item unit to clamp active (0-indexed).
        n_ticks : int
            Number of settling steps per trial.
        store_label : str or None
            If provided, store final hidden activation under this label
            (e.g., 'A' or 'B') for later RSA computation.

        Returns
        -------
        hid_act, out_act : final activations of hidden and output layers
        """
        cfg = self.cfg
        cat_act = torch.zeros(cfg.n_category)
        cat_act[category_idx] = 1.0
        item_act = torch.zeros(cfg.n_item)
        item_act[item_idx] = 1.0

        hid_act = torch.zeros(cfg.n_hidden)
        out_act = torch.zeros(cfg.n_output)

        for t in range(n_ticks):
            hid_act, out_act = self.forward_tick(cat_act, item_act, hid_act, out_act, t)

        self._last_hidden = hid_act.detach().clone()
        self._last_output = out_act.detach().clone()
        self._last_item_act = item_act.detach().clone()
        self._last_cat_act = cat_act.detach().clone()

        if store_label is not None:
            self._pairmate_hidden[store_label] = hid_act.detach().clone()

        return hid_act, out_act

    # ------------------------------------------------------------------
    # Learning update (called after each stimulus presentation)
    # ------------------------------------------------------------------

    def update_weights(self) -> None:
        """
        Apply NMPH learning rule to all modifiable projections,
        using activations from the last presented stimulus.

        Called once per trial (after settling). Implements learning on:
            item → hidden
            category → hidden
            hidden → hidden
            hidden → output
        and their reverses.
        """
        h = self._last_hidden
        o = self._last_output
        it = self._last_item_act
        cat = self._last_cat_act

        with torch.no_grad():
            self.W_item_hid.data = self.learning_rule(self.W_item_hid.data, it, h)
            self.W_hid_item.data = self.learning_rule(self.W_hid_item.data, h, it)
            self.W_cat_hid.data = self.learning_rule(self.W_cat_hid.data, cat, h)
            self.W_hid_cat.data = self.learning_rule(self.W_hid_cat.data, h, cat)
            self.W_hid_hid.data = self.learning_rule(self.W_hid_hid.data, h, h)
            self.W_hid_out.data = self.learning_rule(self.W_hid_out.data, h, o)
            self.W_out_hid.data = self.learning_rule(self.W_out_hid.data, o, h)
            # Keep diagonal of recurrent weights at zero
            self.W_hid_hid.data.fill_diagonal_(0.0)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def pairmate_similarity(self) -> Optional[float]:
        """
        Cosine similarity between stored pairmate A and B hidden representations.
        Returns None if not both are stored.
        """
        if 'A' not in self._pairmate_hidden or 'B' not in self._pairmate_hidden:
            return None
        h_A = self._pairmate_hidden['A']
        h_B = self._pairmate_hidden['B']
        sim = torch.nn.functional.cosine_similarity(h_A.unsqueeze(0), h_B.unsqueeze(0))
        return sim.item()

    def get_rsa_matrix(self) -> torch.Tensor:
        """
        Pairwise cosine similarity matrix of all stored pairmate representations.
        Returns tensor of shape (n_pairmates, n_pairmates).
        """
        labels = sorted(self._pairmate_hidden.keys())
        if not labels:
            return torch.tensor([])
        reps = torch.stack([self._pairmate_hidden[l] for l in labels])  # (n, d)
        # Normalise
        norms = reps.norm(dim=1, keepdim=True).clamp(min=1e-8)
        reps_norm = reps / norms
        return reps_norm @ reps_norm.T  # (n, n)

    def reset_pairmate_store(self) -> None:
        self._pairmate_hidden = {}

    def run_pairmate_simulation(
        self,
        n_ticks: int = 20,
        n_competition_trials: int = 3,
    ) -> dict:
        """
        Standard two-pairmate competition simulation (validates qualitative behaviour).

        Sequence:
            1. Present pairmate A (store hidden rep)
            2. Present pairmate B (competition; store hidden rep; apply learning)
            3. Repeat B presentation (competition continues; track similarity)

        Returns
        -------
        results : dict with keys:
            'similarity_before' : cosine sim before any learning
            'similarity_after'  : cosine sim after competition
            'trajectory'        : list of similarity values at each trial
        """
        # Baseline: present both without learning
        self.present_stimulus(0, 0, n_ticks, store_label='A')
        self.present_stimulus(0, 1, n_ticks, store_label='B')
        sim_before = self.pairmate_similarity()

        trajectory = [sim_before]

        # Competition trials: present B repeatedly (A is competitor)
        for _ in range(n_competition_trials):
            self.present_stimulus(0, 0, n_ticks, store_label='A')
            self.update_weights()
            self.present_stimulus(0, 1, n_ticks, store_label='B')
            self.update_weights()
            trajectory.append(self.pairmate_similarity())

        sim_after = trajectory[-1]

        return {
            'similarity_before': sim_before,
            'similarity_after': sim_after,
            'trajectory': trajectory,
            'direction': 'differentiation' if sim_after < sim_before else 'integration'
                         if sim_after > sim_before else 'no_change',
        }
