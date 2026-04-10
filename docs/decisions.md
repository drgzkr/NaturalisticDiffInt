# NaturalisticDiffInt — Design Decision Log

Append-only log of key design decisions, with rationale. Never delete entries — revise by adding a new entry that supersedes an old one, noting the date and the earlier entry it replaces.

---

## 2026-04-10 | D1: Reimplement from scratch in PyTorch, not port Emergent

**Decision:** Do not attempt to use or port the original Ritvo et al. Go/Emergent code. Reimplement the core mechanisms (kWTA, oscillatory inhibition, BCM/NMPH update rule) in PyTorch from scratch.

**Rationale:**
1. The Emergent framework (`github.com/PrincetonCompMemLab/neurodiff_simulations`) is written in Go, incompatible with the Python/PyTorch stack used across all Dora's projects.
2. The model is small and the mechanisms are clearly documented in the paper and in O'Reilly & Munakata (2000). A from-scratch implementation is feasible in a few hundred lines.
3. Building from scratch gives full control over how the mechanisms are parametrised — essential for the naturalistic extension, which requires modifications not contemplated in the original code.
4. We do not need numerical exact reproduction of Emergent's internal conventions (conductance models, rate-coded spiking, etc.); we need the *functional equivalents*.

**Risk:** Our reimplementation may behave slightly differently from the original, particularly in edge cases of the kWTA dynamics. Mitigation: validate qualitatively against the paper's reported patterns (differentiation in moderate-competitor condition, integration in high-competitor condition) before using in the naturalistic pipeline.

---

## 2026-04-10 | D2: Start with BCM as the U-shaped learning rule

**Decision:** Use the BCM (Bienenstock-Cooper-Munro) formulation as the learning rule, rather than the more complex piecewise-linear coactivity function used in the paper.

**BCM rule:**
```
δw(x→y) = η * y * (y − θ_M) * x
θ_M updated as: θ_M ← τ * θ_M + (1−τ) * y²
```

**Rationale:**
1. BCM is the theoretical foundation the paper explicitly cites (Bienenstock et al. 1982; Cooper 2004). The paper's coactivity-based function is an operationalisation of the same U-shaped principle.
2. BCM has a clear interpretation: θ_M is the sliding modification threshold; activity above it strengthens, below it weakens.
3. Several PyTorch BCM implementations already exist, providing a validation reference.
4. The piecewise-linear function used in the paper requires per-projection parameter fitting by hand, which does not generalise well to the naturalistic setting.

**Revisit condition:** If BCM fails to reproduce qualitative differentiation/integration patterns in the toy validation (D1 risk), switch to the paper's explicit parameterisation.

---

## 2026-04-10 | D3: Pilot with VGG-19 features from StudyForrest

**Decision:** Use VGG-19 pool5 features (4096-dim, TR-sampled) from the StudyForrest dataset as the first input modality to pilot.

**Rationale:**
1. VGG-19 features from StudyForrest are already extracted and aligned with fMRI TRs (from the GazeAware project pipeline). No new extraction required.
2. StudyForrest has 15 subjects, 8 runs, and the most mature associated fMRI RSA analysis infrastructure in Dora's work.
3. The GazeAware paper's encoding models provide a natural prior on which VGG-19 layers carry the most brain-relevant information (pool5 being the best overall predictor for higher visual areas).

**Alternative input to add later:** Qwen-VL 2048-dim embeddings (richer, multimodal, already in NEMM). These will be the second input to test once the pipeline is validated.

---

## 2026-04-10 | D4: Use GSBS-segmented event boundaries to define memory traces

**Decision:** Use GSBS event boundaries (computed on StudyForrest neural data, or on the feature timeseries itself) to segment the input into discrete event traces. Each event trace = the mean feature vector over all TRs within the event.

**Rationale:**
1. GSBS is already implemented and used across Dora's projects. It provides data-driven, principled segmentation.
2. Using neurally-derived boundaries (from GSBS on fMRI data) for defining events is the standard in the field; using feature-derived boundaries is also valid and provides a model-only condition.
3. Event-averaged embeddings are more stable than TR-level embeddings (less noise), making pairmate competition easier to interpret.

**Alternative:** Use the NEMM MemoryArchive's similarity-threshold segmentation. This would create a fully end-to-end model-based pipeline. Try this as a second condition.

---

## 2026-04-10 | D5: Pairmate definition via semantic feature cosine similarity

**Decision:** Define two events as "pairmates" if their cosine similarity in the semantic feature subspace (top-k PCA components of VGG-19 fc7, or LM embedding) exceeds a threshold (θ_pair = 0.7, to be cross-validated), but they occur at least N TRs apart (N ≥ 20, i.e. not adjacent events).

**Rationale:**
1. The key constraint for NMPH competition is *shared features* (overlap in category/item layers). In naturalistic data, "shared features" means events that share scene-level or narrative-level context.
2. Temporal separation ensures we are not just capturing within-scene autocorrelation.
3. The θ_pair threshold mirrors the similarity manipulations in the original experiments (Chanales 2021: high vs low color similarity; Favila 2016: same vs different face associate).

**Uncertainty:** The right θ_pair is unknown. Plan: run a sensitivity analysis sweeping θ_pair ∈ {0.5, 0.6, 0.7, 0.8} and assess which produces the most meaningful differentiation dynamics.

---

## 2026-04-10 | D6: Category vs item feature split via PCA frequency decomposition

**Decision:** Decompose VGG-19 pool5 feature timeseries via PCA. Top K components (slow-varying, high-variance dimensions) = "category features". Bottom components (fast-varying, lower-variance) = "item features". K to be determined empirically (start with K = 20).

**Rationale:**
1. In naturalistic video, slowly-varying dimensions track scene context, character identity, and narrative arc — these are the "shared category" features that define pairmates.
2. Rapidly-varying dimensions track moment-to-moment perceptual details — these are the "unique item" features.
3. This maps the narrative structure of the film onto the model's category/item distinction without requiring manual annotation.

**Alternative:** Use separate feature streams — e.g. VGG-19 for item, GPT-2 sentence embeddings for category. More principled but requires additional preprocessing.
