# NaturalisticDiffInt

**Naturalistic Differentiation–Integration: NMPH-inspired memory dynamics in continuous neural representations**

This project adapts the Ritvo et al. (2024) neural network model of memory differentiation and integration — originally designed for discrete pairmate paradigms — to operate over continuous, naturalistic input timeseries (CNN feature trajectories, multimodal embedding sequences) and to generate predictions that can be compared with brain data from naturalistic fMRI.

---

## Core scientific question

Ritvo et al. (2024) show that a simple unsupervised, U-shaped (BCM-like) learning rule acting on competing memories accounts for whether neural representations become more similar (integrate) or more distinct (differentiate), depending on how strongly an inactive competitor is reactivated. The key mechanism — competitor activity level determining the direction of representational change — is general enough that it should operate over *any* sufficiently rich representational substrate, not just the binary pairmate stimuli in the original model.

**This project asks:** What happens when the same NMPH learning principle is applied to a continuous stream of rich, high-dimensional representations — the kind extracted from a naturalistic video? Can we observe emergent differentiation and integration dynamics in the memory trace space, and do those dynamics predict patterns in fMRI representational geometry?

---

## Relationship to existing projects

| Project | Relationship |
|---|---|
| **NEMM** (`~/Projects/NEMM/`) | Parallel — NEMM segments events via cosine-threshold gating; this project models *how* event memory traces then compete and change relationally |
| **GazeAware** | Input features: VGG-19 (or gaze-conditioned) CNN timeseries from StudyForrest are a natural input here |
| **GlobalStatesMemory** | Brain comparison target: hippocampal RSA and global state modulation of encoding; NMPH predictions map naturally onto hippocampal RSA change |
| **MetastableReview** | Theoretical bridge: NMPH differentiation → anticorrelation; anticorrelated representations are a core MetastableReview theme |

---

## Repository structure

```
NaturalisticDiffInt/
├── README.md                    ← this file
├── JOURNAL.md                   ← session journal (append-only log)
├── docs/
│   ├── architecture.md          ← working model architecture notes
│   ├── hypotheses.md            ← formal hypothesis register
│   └── decisions.md             ← key design decisions with rationale
├── literature/
│   ├── notes/
│   │   └── ritvo2024_nmph.md    ← detailed paper note
│   └── themes/
│       └── nmph_naturalistic.md ← thematic synthesis note
├── src/
│   ├── core/                    ← NMPH model implementation
│   │   ├── __init__.py
│   │   ├── network.py           ← NMPHNetwork: layers, projections
│   │   ├── learning.py          ← U-shaped learning rule (BCM-style)
│   │   ├── inhibition.py        ← kWTA + oscillatory inhibition
│   │   └── memory_archive.py    ← manages competing memory traces
│   ├── input/                   ← input preprocessing
│   │   ├── __init__.py
│   │   ├── cnn_features.py      ← load VGG-19 / Qwen-VL feature timeseries
│   │   └── multimodal.py        ← multimodal input assembly
│   ├── analysis/                ← RSA and brain comparison
│   │   ├── __init__.py
│   │   ├── rsa.py               ← representational similarity analysis
│   │   └── brain_comparison.py  ← compare model RSA with fMRI RSA
│   └── utils/
│       ├── __init__.py
│       └── visualisation.py     ← standard plots
├── notebooks/
│   ├── exploratory/             ← free-form experiments
│   └── analysis/                ← structured analyses
├── data/
│   ├── raw/                     ← input feature timeseries (git-ignored)
│   └── processed/               ← cached intermediate tensors (git-ignored)
├── results/
│   ├── figures/
│   └── tables/
└── journal/                     ← dated session notes (markdown)
    └── 2026-04-10_session1.md
```

---

## Upstream code

The original Ritvo et al. (2024) model is implemented in **Go** using the **Emergent** simulation framework (Leabra algorithm):

- Repository: `https://github.com/PrincetonCompMemLab/neurodiff_simulations`
- Archive: Nguyen 2024 (cited in paper)
- Status: **not directly reusable** — Go/Emergent is incompatible with the Python/PyTorch stack; a PyTorch reimplementation is required

See `docs/architecture.md` for the reimplementation plan.

---

## Input modalities planned

| Modality | Source | Notes |
|---|---|---|
| CNN (VGG-19) features | StudyForrest frames | Already extracted for GazeAware paper |
| Qwen-VL multimodal embeddings | NEMM pipeline | 2048-dim, TR-sampled |
| Language model features | Transcribed audio | GPT-2/LLaMA sentence embeddings |
| Gaze-weighted CNN features | GazeAware pipeline | Gaze-conditioned spatial features |

---

## Target brain comparisons

- **RSA in hippocampus** (StudyForrest): Does model-derived representational change (differentiation/integration of event traces) correlate with hippocampal pattern similarity change across events?
- **Global brain state modulation** (GlobalStatesMemory link): Is differentiation more likely in TPN-dominant states (higher inhibitory tone) than in DMN-dominant states?
- **Temporal receptive window structure**: Do regions with short TRWs show more integration (strong competitor reactivation), and regions with long TRWs show more differentiation?
- **Anti-correlated representation predictions**: Does model-predicted anticorrelation between event traces match fMRI anti-similarity patterns between events?

---

## Colab notebooks

All analysis is designed to run in Google Colab with data on Google Drive.
Each notebook syncs to this GitHub repo via the standard NEMM-style GitHub Sync cells
(requires a `GITHUB_TOKEN` PAT in Colab Secrets).

| Notebook | Purpose | Open |
|---|---|---|
| **01_nmph_toy_validation** | Validate NMPH dynamics: sweep `osc_amp`, reproduce differentiation/integration U-shape, BCM vs piecewise rule comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/NaturalisticDiffInt/blob/main/notebooks/exploratory/01_nmph_toy_validation.ipynb) |
| **02_naturalistic_pipeline** | Full naturalistic pipeline: load VGG-19 features from Drive, PCA split, run MemoryArchive over StudyForrest, produce RSA change matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/NaturalisticDiffInt/blob/main/notebooks/analysis/02_naturalistic_pipeline.ipynb) |
| **03_brain_comparison** | Compare model RSA change with fMRI RSA (Schaefer 400 parcels), test H1 (hippocampal RSA decrease) and H4 (global brain state modulation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/NaturalisticDiffInt/blob/main/notebooks/analysis/03_brain_comparison.ipynb) |

### Google Drive layout expected

```
MyDrive/NaturalisticDiffInt/
├── features/
│   └── vgg19_pool5_run{1-8}.npy        # (n_trs, 4096) float32 — from GazeAware pipeline
├── boundaries/
│   ├── gsbs_boundaries_run{1-8}.npy    # (n_events,) int — TR onsets  [optional]
│   └── gsbs_durations_run{1-8}.npy     # (n_events,) int — TR durations [optional]
├── neural/
│   ├── bold_schaefer400_run{1-8}.npy   # (n_trs, 400) float32 — z-scored parcellated BOLD
│   └── global_state_run{1-8}.npy       # (n_trs,) float32 — TPN dominance score [optional, for H4]
└── results/                             # output directory — created automatically
```

If GSBS boundaries are not provided, notebook 02 falls back to fixed-window segmentation (15 TRs).
If global state timeseries are not provided, H4 (global brain state test) is skipped in notebook 03.

---

## Quick start (local)

```bash
cd ~/Projects/NaturalisticDiffInt
pip install -e .   # once setup.py / pyproject.toml is in place
python src/core/network.py --demo   # sanity check
```
