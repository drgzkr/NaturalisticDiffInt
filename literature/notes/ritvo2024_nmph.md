---
title: "A neural network model of differentiation and integration of competing memories"
authors: Ritvo, Nguyen, Turk-Browne, Norman
year: 2024
journal: eLife
doi: 10.7554/eLife.88608
zotero_key: AZHACXUL
cite_key: ritvo2024_nmph
relevance: high
tags: [NMPH, BCM, unsupervised-learning, differentiation, integration, hippocampus, RSA, computational-model, pairmate-competition, anticorrelation]
source: full-text
---

## Core Question

Can a purely unsupervised, U-shaped learning rule acting on competing memories account for the divergent patterns of representational differentiation and integration observed across a wide range of fMRI experiments?

## Summary

Ritvo et al. present a neural network model implementing the **Nonmonotonic Plasticity Hypothesis (NMPH)** — the idea that the direction of synaptic change (strengthening vs. weakening) depends on the level of activity in a competing memory during retrieval of a target memory. The model uses no supervised learning; instead, weight changes are driven by the coactivity of pre- and post-synaptic units through a U-shaped function. When a competitor memory is inactive, no change occurs. When it is moderately active, shared connections are weakened (differentiation). When it is highly active, shared connections are strengthened (integration). The model is implemented in Emergent (Go; Leabra algorithm) and successfully accounts for findings from three published experiments: colour memory repulsion (Chanales et al. 2021), scene pairmate differentiation (Favila et al. 2016), and curriculum-dependent representational change (Schlichting et al. 2015). Three key novel predictions emerge: differentiation is rapid and abrupt, representational change is asymmetric (later-encoded item distorts), and differentiation produces anticorrelated representations in the source region.

## Methods

**Network architecture:** Four layers — category (shared features), item (unique features), hidden (internal representation; 10-20 units), output (associated stimuli). All projections bidirectional and fully connected; hidden→hidden recurrent. All projections have weak random initial weights plus structured "pre-wired" strong connections for each pairmate.

**Inhibitory dynamics:** k-Winners-Take-All (kWTA) with a `Target Diff` extension (allows "tied" units near the k-th rank to activate). Sinusoidal oscillatory inhibition (amplitude `osc_amp`; varies per model version) lowers inhibition periodically, allowing competitor units to "pop up."

**Learning:** U-shaped coactivity function; parameters differ per projection and per experiment version. Fit by hand (automatic optimisation deemed infeasible due to the highly nonlinear, interactive dynamics).

**Simulations:** Three experiment versions, each adapting the architecture to manipulate competitor activity differently — via stimulus similarity overlap (Chanales), shared associate presence (Favila), and oscillation amplitude × learning curriculum (Schlichting).

**Code availability:** `https://github.com/PrincetonCompMemLab/neurodiff_simulations` (Go; archived as Nguyen 2024).

## Key Findings

1. The model differentiates pairmate representations when competitors are moderately active and integrates them when competitors are highly active — a direct consequence of the U-shaped learning rule.
2. Differentiation occurs rapidly (within one trial of competition) and is highly sensitive to activity dynamics; small changes in oscillation amplitude can flip the outcome from differentiation to integration.
3. Differentiation is asymmetric: pairmate 2 (later-encoded, which "pops up" as competitor) distorts; pairmate 1 (first-encoded, anchor) stays stable.
4. Differentiation manifests as *anticorrelation* in the hidden layer — representations do not merely become less similar; they become opposed.
5. Integration can be symmetric or asymmetric, depending on which pairmate reaches the high-activity zone first.
6. Differentiation is favoured in hippocampus (sparse coding, high learning rate); integration is more likely in neocortex (slower learning rate, which causes "relapse" — shared units get reabsorbed).
7. The model qualitatively fits all three target experiments using different methods to manipulate competitor activity.

## Limitations

- Parameters fitted by hand; no quantitative fit or uncertainty bounds reported. The parameter space is large and interactions are nonlinear, making it hard to know how robust the fits are.
- Model intentionally omits supervised/error-driven learning; the paper acknowledges this is a partial account.
- Very small networks (10–20 hidden units); generalisation to rich, high-dimensional feature spaces not tested.
- Anticorrelation prediction may not scale from single-neuron to BOLD; authors acknowledge this explicitly.
- No temporal dynamics beyond the single competition event are modelled (no sequential episodic learning over a continuous stream).
- "Pairmates" are always a pair; not a memory population. Competition structure in naturalistic paradigms is more complex (many possible competitors per event).
- Code is in Go/Emergent, requiring reimplementation for Python-based workflows.

## Relevance to Dora's Work

**NaturalisticDiffInt (this project):** The entire theoretical and computational basis. The core goal is to extend this model to naturalistic data.

**MetastableReview:** The paper's prediction that differentiation yields *anticorrelated* representations is deeply resonant with FlippingPaper's main finding that anti-correlated neural states dominate naturalistic brain dynamics. The NMPH may be a mechanistic explanation for *why* anti-correlated representational geometries emerge in hippocampus during naturalistic viewing.

**GlobalStatesMemory:** The model's key parameter — oscillation amplitude, which controls whether differentiation or integration occurs — is plausibly modulated by global brain state (TPN/DMN balance). TPN states suppress competitors (favouring differentiation); DMN states allow reactivation (favouring integration). This is H4 in the hypothesis register.

**NEMM:** The NEMM MemoryArchive performs cosine-similarity-gated event segmentation — it defines event traces. NaturalisticDiffInt is the downstream module that handles *competition between* those traces.

## Connections to Other Papers

→ `franklinEtAl2020_SEM.md` — Franklin et al.'s SEM model also performs online event segmentation; both SEM and NMPH-net operate on event representations, but SEM focuses on segmentation, NMPH-net on post-encoding competition.  
→ `baldassano2017_HMM.md` — Baldassano's HMM model identifies event boundaries in fMRI; these boundaries are where NMPH competition likely initiates.  
→ `favila2016_differentiation.md` (if noted) — one of the three experiments modelled.  
→ `chanales2021_color.md` (if noted) — one of the three experiments modelled; colour memory repulsion as behavioural proxy for differentiation.  
→ `schlichting2015_curriculum.md` (if noted) — blocked vs interleaved learning and curriculum effects on differentiation.

## Key Quotes

> "connections to moderately active competitors are weakened (leading to differentiation), and connections to highly active competitors are strengthened (leading to integration)" (Abstract)

> "the way that it actually shows up on individual model runs is as anticorrelation between pairmates; in the model, the size of the aggregate differentiation effect is determined by the proportion of model runs that show this anticorrelation effect" (p.29)

> "differentiation must happen quickly (or it will not happen at all)" (Discussion, p.27)

> "Anything that affects how strongly the competitor comes to mind can impact the kinds of representational change that are observed." (p.27)
