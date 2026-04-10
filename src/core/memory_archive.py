"""
MemoryArchive — naturalistic extension of the NMPH model.

While NMPHNetwork implements the toy two-pairmate simulation,
MemoryArchive manages a growing collection of event-level memory traces
extracted from a continuous naturalistic stimulus stream (e.g. a movie).

Each event trace is a vector in a compressed feature space.
When a new event is processed, the archive:
    1. Identifies competitors (prior events with high semantic similarity)
    2. Runs an oscillatory retrieval pass (simulating competitor reactivation)
    3. Applies NMPH weight updates (modifies the trace's hidden representation)
    4. Tracks the representational similarity matrix across all traces

This is not present in the Ritvo et al. (2024) model, which operates on
pre-defined pairmate pairs. This module is the key design contribution of
NaturalisticDiffInt.

Design decisions implemented here:
    D3: VGG-19 features as input
    D4: GSBS-segmented event boundaries
    D5: Pairmate definition via semantic cosine similarity
    D6: Category/item split via PCA frequency decomposition
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryTrace:
    """
    One segmented event trace — the naturalistic analogue of a pairmate.

    Attributes
    ----------
    idx : int
        Temporal index in the event sequence (0 = earliest).
    hidden : torch.Tensor, shape (n_hidden,)
        Current hidden-layer representation (updated by NMPH learning).
    hidden_init : torch.Tensor, shape (n_hidden,)
        Hidden representation at encoding time (frozen snapshot for RSA change measurement).
    category_features : torch.Tensor, shape (n_category,)
        Slow-varying, semantically rich features (used for competitor selection).
    item_features : torch.Tensor, shape (n_item,)
        Fast-varying, perceptual features (used for learning updates).
    n_trs : int
        Number of fMRI TRs this event spans.
    onset_tr : int
        TR index at event onset.
    competitor_idxs : list of int
        Indices of events identified as pairmates (filled during competition).
    sim_change : float or None
        Cosine similarity change relative to hidden_init after NMPH update.
    """
    idx: int
    hidden: torch.Tensor
    hidden_init: torch.Tensor
    category_features: torch.Tensor
    item_features: torch.Tensor
    n_trs: int = 1
    onset_tr: int = 0
    competitor_idxs: list = field(default_factory=list)
    sim_change: Optional[float] = None

    def representational_change(self) -> float:
        """Cosine similarity of current hidden vs initial hidden."""
        sim = F.cosine_similarity(
            self.hidden.unsqueeze(0),
            self.hidden_init.unsqueeze(0)
        ).item()
        self.sim_change = sim
        return sim


class MemoryArchive:
    """
    Manages a growing collection of event-level memory traces.
    Runs pairmate competition and NMPH-style learning over a naturalistic event stream.

    Parameters
    ----------
    n_hidden : int
        Dimensionality of the hidden representation (after projection from features).
    competitor_threshold : float
        Cosine similarity threshold in category-feature space above which
        two events are considered pairmates.
    min_temporal_gap : int
        Minimum number of events between two traces for them to be considered pairmates
        (prevents adjacent events from being treated as competitors).
    osc_amp : float
        Amplitude of inhibitory oscillation during competitor retrieval.
        Higher → more competitor activation → more integration.
        Lower → less competitor activation → more differentiation.
    osc_period : int
        Period of oscillation in ticks.
    n_osc_ticks : int
        Number of ticks in one competitor retrieval pass.
    learning_rule : str
        'bcm' or 'nmph'. Determines which learning rule is applied.
    lr : float
        Learning rate.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        competitor_threshold: float = 0.7,
        min_temporal_gap: int = 3,
        osc_amp: float = 0.11,
        osc_period: int = 10,
        n_osc_ticks: int = 20,
        learning_rule: str = 'bcm',
        lr: float = 0.01,
    ):
        self.n_hidden = n_hidden
        self.competitor_threshold = competitor_threshold
        self.min_temporal_gap = min_temporal_gap
        self.osc_amp = osc_amp
        self.osc_period = osc_period
        self.n_osc_ticks = n_osc_ticks
        self.lr = lr

        self.traces: list[MemoryTrace] = []

        # Projection layers: feature space → hidden (n_hidden)
        # These are initialised lazily on first add_trace call
        self._category_proj: Optional[torch.Tensor] = None  # (n_hidden, n_category)
        self._item_proj: Optional[torch.Tensor] = None      # (n_hidden, n_item)

        # Learning rule
        from .learning import BCMLearningRule, NMPHLearningRule
        if learning_rule == 'bcm':
            self._lr_rule = BCMLearningRule(lr=lr)
        else:
            self._lr_rule = NMPHLearningRule()

        # RSA log: list of (event_pair, sim_before, sim_after, direction) tuples
        self.competition_log: list[dict] = []

    # ------------------------------------------------------------------
    # Projection layer initialisation
    # ------------------------------------------------------------------

    def _init_projections(self, n_category: int, n_item: int) -> None:
        """Initialise random projection matrices (Gaussian, normalised columns)."""
        self._category_proj = torch.randn(self.n_hidden, n_category)
        self._category_proj /= self._category_proj.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self._item_proj = torch.randn(self.n_hidden, n_item)
        self._item_proj /= self._item_proj.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def _project(
        self,
        category_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project concatenated features to n_hidden hidden representation."""
        if self._category_proj is None:
            self._init_projections(category_features.shape[0], item_features.shape[0])
        h_cat = self._category_proj @ category_features
        h_item = self._item_proj @ item_features
        h = h_cat + h_item
        # Apply ReLU sparsity as a simple kWTA proxy
        h = torch.relu(h)
        h = F.normalize(h, dim=0)
        return h

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_trace(
        self,
        category_features: torch.Tensor,
        item_features: torch.Tensor,
        n_trs: int = 1,
        onset_tr: int = 0,
    ) -> MemoryTrace:
        """
        Encode a new event as a memory trace.

        Parameters
        ----------
        category_features : (n_category,)
            Semantic/slow features (e.g. top PCA components of VGG-19).
        item_features : (n_item,)
            Perceptual/fast features (e.g. bottom PCA components).
        n_trs : int
            Duration of the event in TRs.
        onset_tr : int
            TR onset of the event.

        Returns
        -------
        trace : MemoryTrace
        """
        h = self._project(category_features, item_features)
        trace = MemoryTrace(
            idx=len(self.traces),
            hidden=h.clone(),
            hidden_init=h.clone(),
            category_features=category_features.clone(),
            item_features=item_features.clone(),
            n_trs=n_trs,
            onset_tr=onset_tr,
        )
        self.traces.append(trace)
        return trace

    def find_competitors(self, trace: MemoryTrace) -> list[MemoryTrace]:
        """
        Find all prior traces that are pairmates of the given trace.

        Pairmate criteria (Decision D5):
            - cosine(category_features_i, category_features_j) > competitor_threshold
            - |idx_i - idx_j| >= min_temporal_gap
            - i < j (only look backwards in time)
        """
        competitors = []
        for prior in self.traces:
            if prior.idx >= trace.idx:
                continue
            if (trace.idx - prior.idx) < self.min_temporal_gap:
                continue
            sim = F.cosine_similarity(
                trace.category_features.unsqueeze(0),
                prior.category_features.unsqueeze(0),
            ).item()
            if sim >= self.competitor_threshold:
                competitors.append(prior)
        return competitors

    def _oscillatory_retrieval_pass(
        self,
        target: MemoryTrace,
        competitor: MemoryTrace,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate competitor reactivation via oscillatory inhibition.

        Returns (target_act, competitor_act) — the activity levels at the
        end of the oscillatory pass. These are used as pre/post activities
        for the NMPH learning update.

        Simplified implementation:
            - target is fully active (clamped)
            - competitor activity is driven by their hidden-layer dot product,
              gated by the oscillatory inhibition level
            - oscillation amplitude controls how much competitor activity rises
        """
        import math

        target_act = target.hidden.clone()
        comp_acts = []

        for t in range(self.n_osc_ticks):
            osc_factor = 1.0 - self.osc_amp * math.sin(
                2 * math.pi * t / self.osc_period
            )
            # Competitor excitation = overlap with target
            overlap = torch.dot(target.hidden, competitor.hidden)
            # Competitor activity = overlap / inhibition level
            comp_act_level = overlap / max(osc_factor, 1e-6)
            comp_act_level = torch.clamp(comp_act_level, 0.0, 1.0)
            comp_act = competitor.hidden * comp_act_level
            comp_acts.append(comp_act)

        # Use the mean competitor activity across the oscillatory cycle
        competitor_act = torch.stack(comp_acts).mean(dim=0)
        return target_act, competitor_act

    def run_competition(self, trace: MemoryTrace) -> list[dict]:
        """
        Run pairmate competition for a given trace.
        For each competitor, run an oscillatory retrieval pass and apply NMPH learning.

        Returns
        -------
        results : list of dicts, one per competition episode
        """
        competitors = self.find_competitors(trace)
        trace.competitor_idxs = [c.idx for c in competitors]
        results = []

        for comp in competitors:
            sim_before = F.cosine_similarity(
                trace.hidden.unsqueeze(0), comp.hidden.unsqueeze(0)
            ).item()

            # Retrieval pass
            target_act, competitor_act = self._oscillatory_retrieval_pass(trace, comp)

            # NMPH learning: update both traces' hidden representations
            # Treat hidden vectors as "weights" for a simplified single-layer update
            # target trace is updated based on competition with competitor
            h_target_new = self._lr_rule(
                trace.hidden.unsqueeze(0),   # (1, n_hidden) as "weights"
                competitor_act,               # (n_hidden,) as pre-act
                target_act,                   # (n_hidden,) as post-act
            ).squeeze(0)

            trace.hidden = F.normalize(h_target_new, dim=0)

            sim_after = F.cosine_similarity(
                trace.hidden.unsqueeze(0), comp.hidden.unsqueeze(0)
            ).item()

            direction = (
                'differentiation' if sim_after < sim_before - 0.01
                else 'integration' if sim_after > sim_before + 0.01
                else 'no_change'
            )

            entry = {
                'target_idx': trace.idx,
                'competitor_idx': comp.idx,
                'sim_before': sim_before,
                'sim_after': sim_after,
                'delta_sim': sim_after - sim_before,
                'direction': direction,
                'competitor_activity': competitor_act.norm().item(),
            }
            results.append(entry)
            self.competition_log.append(entry)

        return results

    def process_event_stream(self, category_features: torch.Tensor, item_features: torch.Tensor,
                              onset_trs: list[int] | None = None) -> list[dict]:
        """
        Process a full sequence of events (already segmented).

        Parameters
        ----------
        category_features : (n_events, n_category)
        item_features : (n_events, n_item)
        onset_trs : list of int or None
            TR onset of each event (optional).

        Returns
        -------
        all_results : flat list of competition episode dicts
        """
        n_events = category_features.shape[0]
        all_results = []

        for i in range(n_events):
            onset = onset_trs[i] if onset_trs is not None else i
            trace = self.add_trace(
                category_features[i],
                item_features[i],
                onset_tr=onset,
            )
            results = self.run_competition(trace)
            all_results.extend(results)

        return all_results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_rsa_matrix(self, use_init: bool = False) -> torch.Tensor:
        """
        Pairwise cosine similarity matrix of all event traces.

        Parameters
        ----------
        use_init : bool
            If True, use initial (pre-competition) hidden representations.
            If False, use current (post-competition) representations.

        Returns
        -------
        rsa : (n_events, n_events)
        """
        if not self.traces:
            return torch.tensor([])
        vecs = torch.stack([
            t.hidden_init if use_init else t.hidden
            for t in self.traces
        ])  # (n, d)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        vecs_norm = vecs / norms
        return vecs_norm @ vecs_norm.T

    def get_rsa_change_matrix(self) -> torch.Tensor:
        """RSA_after - RSA_before: positive = integration, negative = differentiation."""
        return self.get_rsa_matrix(use_init=False) - self.get_rsa_matrix(use_init=True)

    def summarise_competition_log(self) -> dict:
        """Return aggregate statistics from all competition episodes."""
        if not self.competition_log:
            return {}
        directions = [e['direction'] for e in self.competition_log]
        delta_sims = [e['delta_sim'] for e in self.competition_log]
        comp_acts = [e['competitor_activity'] for e in self.competition_log]
        return {
            'n_episodes': len(self.competition_log),
            'n_differentiation': directions.count('differentiation'),
            'n_integration': directions.count('integration'),
            'n_no_change': directions.count('no_change'),
            'mean_delta_sim': float(np.mean(delta_sims)),
            'mean_competitor_activity': float(np.mean(comp_acts)),
        }
