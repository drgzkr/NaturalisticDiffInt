---
theme: "NMPH-style memory competition in naturalistic paradigms: bridges, gaps, and adaptation challenges"
papers: [ritvo2024_nmph, franklinEtAl2020_SEM, baldassano2017_HMM]
created: 2026-04-10
updated: 2026-04-10
related_projects: [NaturalisticDiffInt, NEMM, MetastableReview, GlobalStatesMemory]
---

## What these papers share

These papers all address how the brain builds and updates structured representations of continuous, unfolding experience — and specifically, how memory traces from prior events modulate the encoding and retrieval of subsequent ones. They converge on the idea that memory is not a static archive but an active, dynamic structure in which earlier representations compete with, constrain, and shape later ones.

## Points of consensus

- Event-level representations (not frame-level) are the relevant unit for understanding episodic memory structure; neural state transitions mark event boundaries.
- Competition between memory traces — whether expressed as schema prediction error (SEM), boundary detection (HMM), or pairmate competition (NMPH-net) — is central to how the brain manages similar memories.
- Hippocampus is the primary locus of these competitive dynamics, with neocortex playing a slower, downstream role.
- Representational Similarity Analysis (RSA) is the dominant empirical tool for measuring representational change across both lab and naturalistic paradigms.

## Points of tension or debate

- **Learning rule**: Ritvo et al. use an unsupervised U-shaped rule (BCM/NMPH). Franklin et al.'s SEM uses error-driven learning (prediction error on schema transitions). These are not mutually exclusive but occupy different roles in the framework. Ritvo et al. explicitly argue that NMPH *supplements* (not replaces) supervised learning. It is unclear how the two interact in naturalistic viewing.
- **Temporal unit of competition**: SEM/HMM operate at the event level (boundaries define the competition signal). NMPH-net operates at a single-trial level within a controlled lab paradigm. The appropriate timescale for NMPH competition in naturalistic data — within-run, across-run, across-days — is not established.
- **Pairmate definition**: In lab paradigms, pairmates are defined experimentally (same face associate, same scene category). In naturalistic data, this must be derived from feature similarity. There is no consensus on the right level of analysis (scene-level similarity? character-level? narrative-arc similarity?).
- **Direction of generalisation**: Baldassano's HMM and Franklin's SEM were designed with naturalistic data in mind. Ritvo's NMPH-net was designed for lab paradigms. Whether NMPH dynamics are actually observable in naturalistic fMRI — or whether they are masked by the high stimulus dimensionality, the lack of controlled pairmate structure, and the single-exposure design — is entirely unknown.

## How the methods differ

| Aspect | Ritvo 2024 (NMPH-net) | Franklin 2020 (SEM) | Baldassano 2017 (HMM) |
|---|---|---|---|
| Architecture | 4-layer network, pre-wired, ~20 units | RNN + memory store; Holographic Reduced Representations | HMM (no neural network) |
| Learning | Unsupervised BCM/NMPH | Error-driven (prediction error) | None (inference only) |
| Input | Binary vectors (pairmates) | Vector representations of events | BOLD timeseries |
| Temporal scope | Single competition event | Continuous event stream | Continuous fMRI run |
| Output | Representational similarity change | Event boundary predictions | Event boundary timings |
| Competition | Explicit pairmate structure | Schema reuse across similar events | None (state transitions) |
| Brain comparison | RSA in hippocampus | Whole-brain ISC + RSA | Whole-brain boundary detection |
| Code language | Go (Emergent) | Python (JAX/NumPy) | Python (hmmlearn) |

## What they leave open

- **Can NMPH dynamics be observed in naturalistic fMRI?** This is the central open question for NaturalisticDiffInt. No study has directly tested whether the U-shaped competition outcome (differentiation vs. integration direction) can be predicted from naturalistic stimuli and verified in fMRI RSA.
- **What modulates oscillation amplitude (competitor reactivation strength) in naturalistic settings?** The paper suggests inhibitory dynamics in specific regions. Global brain state (TPN/DMN) is an unexplored but plausible modulator. This would link NMPH-net to GlobalStatesMemory.
- **Temporal dynamics of competition over extended naturalistic exposure**: The original model acts on a single competition episode. How do NMPH dynamics play out over a full 2-hour film? Do differentiation effects accumulate across runs? Do they depend on narrative arc structure?
- **Biologically motivated pairmate definition for uncontrolled stimuli**: The field lacks a principled method for identifying which events "compete" in a naturalistic stream. This is a prerequisite for testing any NMPH predictions in naturalistic data.
- **Interaction of NMPH with event segmentation**: Do event boundaries (as detected by GSBS or HMM) coincide with competition episodes? If NMPH competition requires retrieving a prior memory while encoding a new one, event boundaries may be the natural trigger point — but this is speculative.
- **Relation to anti-correlated neural dynamics**: FlippingPaper establishes that anti-correlated neural state transitions dominate naturalistic brain dynamics at multiple scales. Ritvo et al. predict that differentiation gives rise to anti-correlated representations. Are the anti-correlations seen in FlippingPaper evidence of ongoing NMPH-style differentiation across the cortex? This would be a theoretically rich connection.

## Synthesis paragraph

Taken together, these papers sketch an emerging but still-incomplete picture of how the brain navigates the tension between remembering what is unique about each episode and generalising across structurally similar ones. Ritvo et al.'s NMPH framework provides the most mechanistically explicit account of how individual memory traces reshape each other: the U-shaped learning rule determines whether competition leads to separation or merger of representations, and the key lever is competitor activity level, which is itself a function of inhibitory dynamics. Franklin et al.'s SEM and Baldassano et al.'s HMM provide complementary accounts of *when* competition occurs (at event boundaries) and *what* is being competed over (event schemas), but neither model addresses what happens to the *relative representational geometry* of competing events — the differentiation/integration question Ritvo et al. tackle directly. The central challenge for NaturalisticDiffInt is to establish whether the NMPH's competition dynamics, so cleanly demonstrated in controlled pairmate paradigms, leave a legible signature in the representational geometry of naturalistic fMRI. The signal, if it exists, is likely in hippocampus (where sparse coding and high learning rate favour differentiation) and may be most clearly expressed at the level of event pairs that share semantic context but differ perceptually — the naturalistic equivalent of the coloured-barn pairmates in Chanales et al. 2021. The modulation of this signal by global brain state (H4) represents a particularly tractable bridge hypothesis, as both the global state timeseries and the hippocampal RSA data are available from StudyForrest and within the GazeAware/GlobalStatesMemory analysis infrastructure.

## Papers (brief)

- **Ritvo et al. 2024** — Implements NMPH in a minimal neural network; demonstrates differentiation and integration as outcomes of a single U-shaped learning rule; predicts anticorrelation and asymmetry of differentiation. The direct theoretical foundation for NaturalisticDiffInt.
- **Franklin et al. 2020 (SEM)** — Neuro-symbolic model of event segmentation using error-driven learning on event schemas; addresses segmentation dynamics in continuous streams. Relevant as a model of *when* competition is triggered.
- **Baldassano et al. 2017 (HMM)** — Probabilistic model of neural event segmentation in naturalistic fMRI; provides the empirical boundary detection infrastructure that can define event units for the NMPH pipeline.
