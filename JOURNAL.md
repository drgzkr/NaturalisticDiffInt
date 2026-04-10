# NaturalisticDiffInt — Session Journal

An append-only log of exploration sessions. Most recent at top. Each entry is a session or a focused thinking block. Not a polished document — write raw.

---

## 2026-04-10 — Session 0: Project inception

**Context:** Dora raised the idea of adapting Ritvo et al. (2024) to naturalistic data — specifically, applying the NMPH (Nonmonotonic Plasticity Hypothesis) learning dynamics to continuous, rich feature timeseries (CNN/multimodal embeddings) extracted from video, and then comparing the resulting memory dynamics with brain data.

**Paper read:** Ritvo, Nguyen, Turk-Browne, Norman (2024). *A neural network model of differentiation and integration of competing memories.* eLife 12:RP88608.

### What the original model does

The Ritvo model (hereafter NMPH-net) is a very small, handcrafted network built in **Emergent** (Go). It has:
- Four layers: **category** (shared features), **item** (unique features), **hidden** (internal rep), **output** (associates/consequences)
- Bidirectional connections between all layers; hidden→hidden recurrent connections
- **kWTA** (k-Winners-Take-All) inhibitory dynamics, with a sinusoidal oscillation that periodically lowers inhibition, allowing competitor memories to "pop up"
- A **U-shaped (BCM-style) learning rule**: when pairmate B is retrieved, pairmate A can become active as a competitor. If A is inactive → no change. If A is *moderately* active → connections to shared units **weakened** (differentiation). If A is *highly* active → connections to shared units **strengthened** (integration).
- Pre-wired connections encode initial memory strength; learning then modulates these

The model was used to simulate 3 experiments (Chanales 2021: color repulsion; Favila 2016: scene pairmates; Schlichting 2015: blocked vs interleaved learning). It makes three key novel predictions:
1. Differentiation is **rapid and abrupt** (occurs after first competition moment)
2. Differentiation is **asymmetric** (the later-encoded item distorts, the first-encoded item anchors)
3. Differentiation produces **anticorrelated** representations in the source region

### Code availability

- Available at: `https://github.com/PrincetonCompMemLab/neurodiff_simulations`
- Language: **Go** (Emergent framework; Leabra algorithm)
- **Not directly usable** in our Python/PyTorch stack. A reimplementation is required.
- Assessment: the core mechanisms (kWTA, BCM learning, oscillatory inhibition) are well-documented in the paper and are feasible to reimplement in PyTorch. The model is small and not computationally demanding.

### Key conceptual translation challenges

Moving from discrete pairmate paradigms → naturalistic continuous timeseries:

| Original (Ritvo) | Naturalistic adaptation | Challenge |
|---|---|---|
| 2 discrete pairmates (binary vectors) | Many event-trace embeddings (dense, high-dim) | Defining "pairmates" in continuous space |
| Pre-wired connections per item | Online memory archive, TR-sampled | How do memories form? Threshold? Segmentation? |
| Simple 4-layer network, ~10-20 units | Feature space is 2048-dim (Qwen) or 4096-dim (VGG-19) | Projection / compression required |
| Controlled lab experiment | Unsegmented video stream | Need event segmentation layer upstream |
| Single learning episode, few trials | Minutes of video at 0.5 Hz | Timescale and learning rate issues |
| Competitor = a single known pairmate | Competitor = any prior event trace | Need similarity-based competitor selection |

### Architectural decisions to explore

1. **Compression layer**: Project CNN/multimodal features into a manageable hidden representation (e.g. PCA → 50-200 dims, or learned linear projection). This is the "item layer" analogue.
2. **Event segmentation interface**: Feed in pre-segmented event traces from NEMM, or use similarity thresholding inline. Each event trace becomes a "memory" that can compete with subsequent ones.
3. **Pairmate definition**: Two event traces are "pairmates" if they share a high-level context (same scene type, same character, same narrative arc). Operationally: cosine similarity above some threshold θ in a *category* feature space (e.g., higher-level semantic embedding) but different in lower-level perceptual details.
4. **kWTA in high-dim space**: Standard kWTA limits activity to k units. In a continuous embedding space, analogous to top-k activation or sparse coding. Can be approximated with a differentiable sparse activation (e.g., entmax, ReLU + normalisation).
5. **Oscillatory inhibition**: Simplest implementation: multiply inhibition by a sinusoidal envelope during the "retrieval" pass. Period and amplitude are hyperparameters.
6. **U-shaped learning rule**: BCM rule — weight change = post * (post - θ_M) * pre, where θ_M is a sliding modification threshold. Or: can use the coactivity-based formulation from the paper (product of activations mapped through a U-shaped function).
7. **Competitor selection**: At each time step (or event boundary), determine which prior event traces are competitors. Options: (a) all prior traces within a similarity window; (b) top-k most similar; (c) all traces above cosine threshold.

### Hypotheses to test (draft, to be formalised in docs/hypotheses.md)

H1. **Differentiation predicts hippocampal pattern similarity decrease**: Events whose traces become differentiated in the NMPH-net should show decreasing RSA values in hippocampal BOLD patterns across those events.

H2. **Integration predicts hippocampal pattern similarity increase**: When competitor activity is high (similar event traces in close temporal proximity), integration should predict increasing RSA.

H3. **Anticorrelation at event boundaries**: NMPH-net predicts anticorrelated representations following rapid differentiation; hippocampal RSA should show below-zero cross-event correlations in these cases.

H4. **Global brain state modulates direction**: TPN-dominant global states → higher inhibitory tone → moderate competitor activity → differentiation. DMN-dominant → lower inhibition → higher competitor activity → integration. This links NMPH dynamics to GlobalStatesMemory.

H5. **TRW gradient maps onto oscillation amplitude**: Regions with short TRWs (sensory) should show integration dynamics (low inhibition, high competitor activity); regions with long TRWs (DMN, hippocampus) should show differentiation (high inhibition, moderate competitor activity).

### Next steps

- [ ] Implement a minimal PyTorch NMPH-net: single hidden layer, kWTA, BCM update, oscillations
- [ ] Validate on Ritvo's toy pairmate paradigm (reproduce differentiation and integration qualitatively)
- [ ] Plug in StudyForrest VGG-19 feature timeseries as input (already extracted)
- [ ] Define event-level "pairmate" structure from GSBS segmentation output
- [ ] Run representational similarity analysis on NMPH-net's hidden layer timeseries
- [ ] Compare with hippocampal RSA from StudyForrest fMRI data
- [ ] Explore hyperparameter space: oscillation amplitude, kWTA k, learning rate

### Open questions (not yet resolved)

- What is the right granularity for defining "competing" events in continuous naturalistic data? Scene-level, narrative-arc-level, or semantic-cluster-level?
- Can we define a principled "category" vs "item" layer split for naturalistic features? E.g., higher-level semantic features (category) vs. lower-level perceptual details (item)?
- Does the U-shaped learning rule need to be implemented locally (unit-by-unit) or can it be approximated at the embedding level?
- How do we handle the temporal ordering of naturalistic events — unlike lab paradigms, the "competitor" is not always defined a priori?
- Is there an existing Python/PyTorch implementation of Leabra we can build on? (Check `leabra-torch` or similar.)

---
