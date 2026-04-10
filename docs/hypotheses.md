# NaturalisticDiffInt — Hypothesis Register

Formal register of testable hypotheses. Each entry includes theoretical basis, operationalisation, expected result, and status.

Status codes: `draft` | `formalised` | `tested-pilot` | `tested-full` | `revised` | `rejected`

---

## H1 — Differentiation predicts hippocampal RSA decrease

**Status:** `draft`

**Theoretical basis:** The NMPH predicts that moderately active competitors undergo weakening of shared connections, leading to reduced representational overlap. In hippocampus, which has sparse coding and a high learning rate (McClelland et al. 1995), these conditions are most favourable for differentiation. Ritvo et al. (2024) simulate this in Favila et al. (2016) and predict that linked pairmates show below-chance pattern similarity in hippocampus.

**Naturalistic operationalisation:**
- *Model variable*: NMPH-net hidden-layer pairwise cosine similarity change for event pairs classified as "differentiated" (Δsim < −ε after competition episode)
- *Brain variable*: Hippocampal RSA between fMRI patterns evoked during pairs of similar events (StudyForrest); computed as Pearson r of voxel patterns at event i vs event j, before and after a competition episode
- *Pairmate definition*: Two events with cosine(category_features_i, category_features_j) > 0.7, i.e. semantically similar events separated in time

**Expected result:** Events that the NMPH-net differentiates (Δsim < 0) correspond to hippocampal pairmate pairs where RSA declines between early and late runs of StudyForrest.

**Falsifiability:** If hippocampal RSA *increases* or does not change for NMPH-differentiated pairs, H1 is falsified for this operationalisation.

**Dependencies:** Requires event segmentation (GSBS or NEMM), NMPH-net implementation, hippocampal RSA computation from StudyForrest.

---

## H2 — Integration predicts hippocampal RSA increase

**Status:** `draft`

**Theoretical basis:** High competitor activity (e.g. when two events are very similar and occur close in time) leads to strengthening of shared connections → integration. In hippocampus, this is observed in paradigms with shared associates (Schapiro et al. 2016; Zeithamova et al. 2018).

**Naturalistic operationalisation:**
- *Model variable*: NMPH-net pairs classified as "integrated" (Δsim > +ε)
- *Brain variable*: Hippocampal RSA increase between those event pairs across runs

**Expected result:** NMPH-integrated event pairs show increased cross-run hippocampal RSA.

**Note:** Integration and differentiation may operate at different timescales. Integration may be more visible with a between-run comparison (after repeated exposure to the film); differentiation may manifest within a single viewing.

---

## H3 — Differentiation produces anticorrelated neural representations

**Status:** `draft`

**Theoretical basis:** A key novel prediction of Ritvo et al. (2024): differentiation shows up not merely as reduced similarity but as *anticorrelation* in the source region. This is because NMPH weakening actively reassigns hidden units from shared to unique representations, creating opposing representational patterns.

**Naturalistic operationalisation:**
- *Model variable*: Hidden-layer pairwise cosine similarity < −ε for differentiated pairs
- *Brain variable*: Cross-event hippocampal RSA (voxel-pattern Pearson r) < 0 for those event pairs, contrasted against matched non-competing event pairs

**Expected result:** A subset of events predicted to differentiate by the NMPH-net will show negative-RSA pairs in the DG/CA3 subfield (where differentiation is expected to be strongest; Wammes et al. 2022, Kim et al. 2017).

**Complication:** Anti-correlation at the single-neuron level may not scale to BOLD (Ritvo et al. note this limitation). This may require electrophysiology for definitive test. In fMRI, we may observe reduced (but not negative) RSA.

---

## H4 — Global brain state modulates differentiation/integration balance

**Status:** `draft`

**Theoretical basis:** The Ritvo model shows that oscillation amplitude (which controls how strongly competitors activate) determines whether differentiation or integration occurs. Global brain state (TPN vs DMN dominance) modulates inhibitory tone in hippocampus and cortex — a link proposed in GlobalStatesMemory. TPN states → higher inhibition → moderate competitor activity → differentiation. DMN states → lower inhibition → high competitor activity → integration.

**Naturalistic operationalisation:**
- *Global state variable*: TPN/DMN balance at the moment of event boundary (from FlippingPaper GSBS state time series on StudyForrest)
- *Model variable*: NMPH-net oscillation amplitude is mapped to global brain state: osc_amp ∝ TPN dominance
- *Predicted outcome*: Events occurring during TPN-dominant states should show more differentiation; DMN-dominant states should show more integration

**Expected result:** In fMRI data, hippocampal RSA for similar event pairs decreases more following TPN-state event boundaries, and increases following DMN-state boundaries.

**Link to existing projects:** This hypothesis directly bridges NaturalisticDiffInt with GlobalStatesMemory and FlippingPaper. It could become a joint analysis or a standalone result within GlobalStatesMemory.

---

## H5 — TRW gradient predicts oscillation amplitude gradient (inter-region)

**Status:** `draft`

**Theoretical basis:** Hasson et al. (2015) describe a temporal receptive window (TRW) hierarchy: sensory regions integrate over short timescales, higher association areas over long timescales. Ritvo's model implies that brain regions with lower inhibitory tone (easier competitor reactivation) favour integration; regions with higher inhibitory tone favour differentiation. If TRW reflects integration timescale (long TRW = more integration), then regions with long TRWs should show integration-like RSA dynamics, and regions with short TRWs should show differentiation-like dynamics (if any representational change occurs at all in lower sensory areas).

**Naturalistic operationalisation:**
- *Model variable*: Run NMPH-net with varying oscillation amplitudes; note which regime produces differentiation vs integration
- *Brain variable*: For each Schaefer parcel, compute: (a) estimated TRW (from Hasson ISC windowing or from our own GSBS timescale analysis), (b) cross-event RSA change (integration direction or differentiation direction)
- *Predicted result*: TRW length inversely correlates with likelihood of differentiation; positively correlates with integration

**Note:** This is the most speculative hypothesis. TRW and inhibitory tone are not the same thing; this requires a mechanistic bridge argument.

---

## H6 — Asymmetry of representational change is detectable in naturalistic RSA

**Status:** `draft`

**Theoretical basis:** Ritvo et al. predict that when differentiation occurs, it is *asymmetric*: the later-encoded pairmate distorts, the earlier-encoded anchors. In naturalistic paradigms, "later-encoded" = events occurring later in the film.

**Naturalistic operationalisation:**
- For each pairmate pair (events i < j, same scene type, similar features), compute RSA change for event i and event j separately
- *Predicted result*: RSA change is larger for event j (the later event, the "distorter") than for event i (the earlier event, the "anchor")
- Can also compare: Δsim(i→i) vs Δsim(j→j) across runs (does the later event's representation drift more?)

**Challenge:** Requires within-subject run-to-run RSA tracking and reliable pairing of events by semantic similarity. Feasible with StudyForrest (8 runs per subject).

---

## Null hypotheses to keep honest

- **N1**: NMPH dynamics in the model are unrelated to fMRI RSA change (RSA change is not predicted by any model variable)
- **N2**: Any RSA change observed is fully explained by stimulus-driven similarity structure (no memory competition effect above and beyond initial feature similarity)
- **N3**: Global brain state has no effect on the direction of RSA change (H4 is noise)
