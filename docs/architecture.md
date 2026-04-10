# NaturalisticDiffInt — Architecture Notes

Working notes on the model design. Not final — update freely. Decisions are logged in `docs/decisions.md`.

---

## Original Ritvo model (summary for reimplementation)

Built in **Emergent** (Go, Leabra algorithm). Not reusable directly.

### Layers

| Layer | Role | Size |
|---|---|---|
| Category | Shared perceptual/semantic features; same for pairmates | 1 unit per category |
| Item | Unique item features; differs per pairmate | 1 unit per item |
| Hidden | Internal representation; site of competition and learning | ~10–20 units |
| Output | Associated stimuli / predicted consequences | 1 unit per associate |

All projections are bidirectional and fully connected. Hidden→Hidden recurrent connections also exist.

### Inhibitory dynamics

- **kWTA (k-Winners-Take-All)**: at most k units active per layer. Equivalent to setting inhibition so that units below rank k are suppressed.
- **Target Diff extension**: units within `target_diff` of the k-th unit are allowed to also activate ("tied" units). This implements soft sparsity.
- **Oscillatory inhibition**: inhibition is multiplied by a sinusoidal envelope `1 ± osc_amp * sin(2πt/T)`. When inhibition is low, previously suppressed competitor units can activate ("pop up"). Amplitude `osc_amp` is the main knob controlling differentiation vs. integration.

### Learning rule (NMPH / BCM-style)

Weight change between pre-synaptic unit x and post-synaptic unit y:

```
coactivity(x, y) = act_x * act_y

Δw(x, y) = f_NMPH(coactivity)
```

where `f_NMPH` is a U-shaped function:
- coactivity < θ_low → weakening (differentiation zone)
- coactivity > θ_high → strengthening (integration zone)
- coactivity ≈ 0 → no change (inactive competitor zone)

In practice, this is implemented as a polynomial or piecewise-linear function. Parameters differ per projection.

---

## PyTorch reimplementation plan

### Design principles

1. **Fidelity to mechanisms, not to Emergent API**: Reproduce kWTA, oscillatory inhibition, and NMPH update; do not try to replicate Emergent's internal numerical conventions exactly.
2. **Modular**: keep each mechanism in its own class so we can swap components (e.g., replace kWTA with sparse sigmoid, replace BCM with contrastive Hebbian, etc.).
3. **Vectorised**: support batched processing — we need to run over hundreds of TRs.
4. **Differentiable optional**: the NMPH learning is explicitly *not* backprop-based; use custom `weight_update()` methods rather than PyTorch autograd.

### Core classes

```
src/core/
├── network.py          ← NMPHNetwork (orchestrates forward pass + learning)
├── learning.py         ← NMPHLearningRule (U-shaped coactivity → Δw)
├── inhibition.py       ← KWTALayer (kWTA + oscillation envelope)
└── memory_archive.py   ← MemoryTrace, MemoryArchive (naturalistic extension)
```

#### `NMPHNetwork`

```python
class NMPHNetwork:
    """
    Minimal NMPH network for pairmate competition.
    Inputs: category vector, item vector (binary or continuous)
    Layers: category, item, hidden, output
    Learning: U-shaped coactivity-based weight updates
    """
    def __init__(self, n_hidden, n_output, n_category, n_item, ...)
    def forward(self, category_input, item_input)  # one step
    def oscillate(self, t)                          # update inhibition level at time t
    def update_weights(self)                        # apply NMPH rule
    def compute_rsa(self)                           # cosine similarity between hidden reps
```

#### `NMPHLearningRule`

```python
class NMPHLearningRule:
    """
    U-shaped learning function on coactivity.
    Parameters: theta_low, theta_high, lr_weaken, lr_strengthen
    """
    def delta_w(self, pre_act, post_act) -> torch.Tensor
```

BCM formulation as a simpler alternative:
```
δw_BCM = η * post * (post - θ_M) * pre
θ_M updated as exponential moving average of post² (sliding threshold)
```

#### `KWTALayer`

```python
class KWTALayer:
    """
    k-Winners-Take-All with optional soft Target Diff and sinusoidal oscillation.
    """
    def __init__(self, k, k_max=None, target_diff=0.05, osc_amp=0.0, osc_period=10)
    def apply(self, x, t=0) -> torch.Tensor  # returns sparse activation
```

#### `MemoryArchive` (naturalistic extension)

This is the naturalistic extension — not in the original model. It manages a collection of event-level memory traces and implements the competition mechanics over continuous sequences.

```python
class MemoryTrace:
    """One segmented event trace (analogous to one pairmate in the original)."""
    embedding: Tensor          # current hidden-layer representation
    category_features: Tensor  # higher-level semantic features (for competitor selection)
    item_features: Tensor      # lower-level perceptual details
    timestamp: int
    competitors: List[int]     # indices of pairmates with high category similarity

class MemoryArchive:
    """Manages all memory traces; runs competition and NMPH update."""
    def add_trace(self, embedding, category_features, item_features)
    def find_competitors(self, trace, threshold=0.7)  # cosine on category features
    def run_competition(self, trace_idx)              # oscillation pass + NMPH update
    def get_rsa_matrix(self)                          # pairwise similarity of all traces
```

---

## Naturalistic adaptation: the full pipeline

```
Naturalistic video (StudyForrest, Sherlock, etc.)
         │
         ▼
┌─────────────────────────────────────┐
│  Feature extraction (frozen)         │
│  VGG-19 pool5 → 4096-dim per frame   │
│  or Qwen-VL → 2048-dim per frame     │
│  or LM embedding → 768-dim per TR    │
└──────────────┬──────────────────────┘
               │  TR-sampled timeseries
               ▼
┌─────────────────────────────────────┐
│  Event segmentation (GSBS or NEMM)   │
│  → event boundaries                  │
│  → event-averaged embeddings         │
└──────────────┬──────────────────────┘
               │  event × D tensor
               ▼
┌─────────────────────────────────────┐
│  Feature decomposition              │
│  Category features: PCA / ICA top   │
│    components (shared scene context) │
│  Item features: residual / bottom    │
│    components (unique percepts)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  NMPHNetwork or MemoryArchive       │
│  For each event (in temporal order):│
│    1. Encode → hidden rep           │
│    2. Find competitors (similarity) │
│    3. Oscillatory retrieval pass    │
│    4. NMPH weight update            │
│    5. Log hidden rep post-update    │
└──────────────┬──────────────────────┘
               │  model RSA matrix (events × events)
               ▼
┌─────────────────────────────────────┐
│  Brain comparison                   │
│  fMRI RSA matrix (parcels × events  │
│    × events, from StudyForrest)      │
│  → Spearman correlation per parcel  │
│  → Map back to brain surface         │
└─────────────────────────────────────┘
```

---

## Key hyperparameters and their roles

| Parameter | Ritvo analogue | Role in naturalistic setting |
|---|---|---|
| `osc_amp` | Oscillation amplitude | Controls how strongly competitors activate; main differentiation/integration switch |
| `k` (kWTA) | k in kWTA | Sparsity of hidden representations; higher k → more integration-prone |
| `theta_low`, `theta_high` | U-shape thresholds | Width of the no-change, weakening, strengthening zones |
| `lr_weaken`, `lr_strengthen` | Learning rate per zone | Speed of differentiation / integration |
| `competitor_threshold` | Pairmate definition | Cosine similarity above which two events are defined as pairmates |
| `n_hidden` | Hidden layer size | Bottleneck dimension; controls compression of naturalistic features |
| `osc_period` | Oscillation period | Number of "ticks" per retrieval episode |

---

## Open architectural questions

1. **Category vs item feature split**: In the original model, category and item layers are trivially distinct (1 unit each, pre-assigned). In naturalistic data, this split must be derived. Two options:
   - *Semantic / perceptual split*: use LM features as "category" (semantic context) and VGG-19 mid-layer features as "item" (perceptual details). Rationale: pairmates share narrative context but differ perceptually.
   - *Frequency decomposition*: slow-varying components (low temporal frequency) = category; fast-varying = item. Analogous to the TRW hierarchy.

2. **Temporal order of competition**: In the lab, pairmate A is always encoded before B, so the order of competition is known. In naturalistic data, the "pairmate" is defined post-hoc by similarity. We need to define competition direction by temporal order (earlier trace = anchor, later trace = distorter).

3. **Learning rate in continuous sequences**: Ritvo uses a single learning step per "trial". In continuous data, we need to decide whether to update after every TR, every event boundary, or only when competitors exceed the similarity threshold. The last option is most principled.

4. **Does the hidden layer need to be learned, or can we use the embedding space directly?** Option A (simpler): apply NMPH dynamics directly in the PCA-compressed embedding space. Option B (fuller): learn a projection layer that maps from CNN features to a hidden representation with kWTA-induced sparsity, then apply NMPH.

5. **Multi-subject handling**: fMRI data has inter-subject variability; the model is deterministic given fixed inputs. Solutions: (a) use hyperaligned fMRI data as target; (b) run model per subject using gaze-weighted features; (c) fit model oscillation parameters per subject.

---

## Leabra in Python — alternatives to full reimplementation

- **`leabra-torch`**: Check if a community PyTorch implementation exists (search GitHub). As of 2024 there are partial implementations (`emer/leabra` in Go; `benureau/leabra` in Python — active in 2018, unclear if maintained).
- **PyTorch-BCM**: BCM learning rule (the theoretical basis of NMPH) has several PyTorch implementations. This is a simpler starting point than full Leabra.
- **Recommendation**: Start with a custom BCM + kWTA implementation. It captures the essential dynamics without depending on Leabra's complex full machinery (which includes spike rates, conductance models, etc. not needed here).

---

## References

- Ritvo et al. (2024) eLife — main paper
- O'Reilly & Munakata (2000) *Computational Explorations in Cognitive Neuroscience* — kWTA and Leabra
- Bienenstock, Cooper & Munro (1982) — BCM learning rule
- Norman et al. (2006) *Psychological Review* — NMPH theory
- Detre et al. (2013) — NMPH; Hulbert & Norman (2015)
