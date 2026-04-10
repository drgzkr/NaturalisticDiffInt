"""
RSA (Representational Similarity Analysis) utilities.

Functions for computing, comparing, and visualising representational
similarity matrices from both model and brain data.

Used for:
    - Computing model RSA matrices from MemoryArchive hidden representations
    - Computing fMRI RSA matrices from BOLD timeseries (StudyForrest)
    - Spearman/Pearson correlation between model and brain RSA
    - Cross-run RSA change matrices (integration/differentiation detection)
"""

import numpy as np
import torch
from scipy import stats
from typing import Optional


def cosine_rsa(embeddings: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity matrix.

    Parameters
    ----------
    embeddings : (n_events, n_dims)

    Returns
    -------
    rsa : (n_events, n_events)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalised = embeddings / norms
    return normalised @ normalised.T


def pearson_rsa(timeseries: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson correlation matrix.

    Parameters
    ----------
    timeseries : (n_events, n_voxels_or_dims)

    Returns
    -------
    rsa : (n_events, n_events)
    """
    return np.corrcoef(timeseries)


def upper_triangle(rsa: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract upper triangle (excluding diagonal by default k=1)."""
    return rsa[np.triu_indices_from(rsa, k=k)]


def rsa_correlation(
    model_rsa: np.ndarray,
    brain_rsa: np.ndarray,
    method: str = 'spearman',
    exclude_diagonal: bool = True,
) -> tuple[float, float]:
    """
    Correlate model and brain RSA matrices.

    Parameters
    ----------
    model_rsa, brain_rsa : (n_events, n_events)
    method : 'spearman' or 'pearson'
    exclude_diagonal : bool
        If True, use only upper triangle (standard RSA approach).

    Returns
    -------
    r : float
    p : float
    """
    if exclude_diagonal:
        model_vec = upper_triangle(model_rsa)
        brain_vec = upper_triangle(brain_rsa)
    else:
        model_vec = model_rsa.ravel()
        brain_vec = brain_rsa.ravel()

    if method == 'spearman':
        r, p = stats.spearmanr(model_vec, brain_vec)
    else:
        r, p = stats.pearsonr(model_vec, brain_vec)
    return float(r), float(p)


def rsa_change_matrix(rsa_before: np.ndarray, rsa_after: np.ndarray) -> np.ndarray:
    """
    Compute representational change matrix: RSA_after - RSA_before.

    Positive values = integration (similarity increased).
    Negative values = differentiation (similarity decreased).
    """
    return rsa_after - rsa_before


def event_averaged_fmri(
    bold: np.ndarray,
    event_onsets: np.ndarray,
    event_durations: np.ndarray,
    hrf_delay: int = 4,
) -> np.ndarray:
    """
    Average BOLD signal within each event window to produce event-level representations.

    Parameters
    ----------
    bold : (n_trs, n_voxels)
    event_onsets : (n_events,) in TR units
    event_durations : (n_events,) in TR units
    hrf_delay : int
        Approximate HRF delay in TRs (applied as offset to onset).

    Returns
    -------
    event_reps : (n_events, n_voxels)
    """
    n_events = len(event_onsets)
    n_voxels = bold.shape[1]
    event_reps = np.zeros((n_events, n_voxels))

    for i in range(n_events):
        start = int(event_onsets[i]) + hrf_delay
        end = int(event_onsets[i] + event_durations[i]) + hrf_delay
        start = max(0, min(start, bold.shape[0] - 1))
        end = max(start + 1, min(end, bold.shape[0]))
        event_reps[i] = bold[start:end].mean(axis=0)

    return event_reps


def model_brain_rsa_map(
    model_rsa: np.ndarray,
    bold: np.ndarray,
    event_onsets: np.ndarray,
    event_durations: np.ndarray,
    parcel_labels: Optional[np.ndarray] = None,
    method: str = 'spearman',
) -> dict:
    """
    Compute RSA correlation between model RSA and brain RSA for each parcel.

    Parameters
    ----------
    model_rsa : (n_events, n_events)
    bold : (n_trs, n_parcels)
    event_onsets, event_durations : (n_events,)
    parcel_labels : (n_parcels,) or None
    method : 'spearman' or 'pearson'

    Returns
    -------
    result : dict with keys 'r' (n_parcels,), 'p' (n_parcels,), 'parcel_labels'
    """
    event_reps = event_averaged_fmri(bold, event_onsets, event_durations)
    n_parcels = bold.shape[1]
    r_map = np.zeros(n_parcels)
    p_map = np.zeros(n_parcels)

    for parcel_idx in range(n_parcels):
        brain_rsa = pearson_rsa(event_reps[:, parcel_idx:parcel_idx+1])
        r, p = rsa_correlation(model_rsa, brain_rsa, method=method)
        r_map[parcel_idx] = r
        p_map[parcel_idx] = p

    return {'r': r_map, 'p': p_map, 'parcel_labels': parcel_labels}
