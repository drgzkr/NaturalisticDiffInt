"""
CNN feature loading and preprocessing for naturalistic inputs.

Loads VGG-19 pool5 features (4096-dim, TR-sampled) from the StudyForrest
GazeAware pipeline, and prepares them for use in the NMPH-net / MemoryArchive.

Implements Decision D6: PCA-based category/item feature split.
    - category_features: top K PCA components (slow, semantic)
    - item_features: remaining components (fast, perceptual)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
from sklearn.decomposition import PCA


def load_vgg19_features(
    path: Path | str,
    run: Optional[int] = None,
) -> np.ndarray:
    """
    Load VGG-19 pool5 feature timeseries.

    Expected format: .npy array of shape (n_trs, 4096).
    If run is specified, load `features_run{run}.npy`; else load the path directly.

    Parameters
    ----------
    path : Path or str
        Directory containing feature files or direct path to .npy file.
    run : int or None
        Run index (1-indexed for StudyForrest).

    Returns
    -------
    features : (n_trs, 4096)
    """
    path = Path(path)
    if run is not None:
        fpath = path / f"features_run{run}.npy"
    else:
        fpath = path
    return np.load(fpath).astype(np.float32)


def pca_category_item_split(
    features: np.ndarray,
    n_category: int = 20,
    pca: Optional[PCA] = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Split features into category (slow/semantic) and item (fast/perceptual) subspaces
    via PCA (Decision D6).

    Parameters
    ----------
    features : (n_trs, n_features)
        Feature timeseries.
    n_category : int
        Number of top PCA components assigned to "category" features.
    pca : fitted PCA or None
        If None, fit PCA on `features`. If provided, transform using existing PCA.

    Returns
    -------
    category_features : (n_trs, n_category)
    item_features : (n_trs, n_features - n_category)
    pca : fitted PCA object
    """
    if pca is None:
        pca = PCA()
        projected = pca.fit_transform(features)
    else:
        projected = pca.transform(features)

    category_features = projected[:, :n_category]
    item_features = projected[:, n_category:]
    return category_features, item_features, pca


def event_average_features(
    features: np.ndarray,
    event_onsets: np.ndarray,
    event_durations: np.ndarray,
) -> np.ndarray:
    """
    Compute event-averaged feature vectors (mean over TRs within each event).

    Parameters
    ----------
    features : (n_trs, n_features)
    event_onsets : (n_events,) in TR units (0-indexed)
    event_durations : (n_events,) in TR units

    Returns
    -------
    event_features : (n_events, n_features)
    """
    n_events = len(event_onsets)
    event_features = np.zeros((n_events, features.shape[1]), dtype=np.float32)
    for i in range(n_events):
        start = int(event_onsets[i])
        end = int(event_onsets[i] + event_durations[i])
        start = max(0, min(start, features.shape[0] - 1))
        end = max(start + 1, min(end, features.shape[0]))
        event_features[i] = features[start:end].mean(axis=0)
    return event_features


def prepare_archive_inputs(
    features: np.ndarray,
    event_onsets: np.ndarray,
    event_durations: np.ndarray,
    n_category: int = 20,
    pca: Optional[PCA] = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, PCA]:
    """
    Full preprocessing pipeline: feature loading → PCA split → event averaging.

    Returns
    -------
    category_features_t : (n_events, n_category) as torch.Tensor
    item_features_t : (n_events, n_item) as torch.Tensor
    event_onsets : (n_events,) as np.ndarray (passed through)
    pca : fitted PCA
    """
    # Event-average the raw features
    event_feats = event_average_features(features, event_onsets, event_durations)

    # PCA split
    cat_feats, item_feats, pca = pca_category_item_split(event_feats, n_category, pca)

    # Normalise
    cat_norm = cat_feats / (np.linalg.norm(cat_feats, axis=1, keepdims=True) + 1e-8)
    item_norm = item_feats / (np.linalg.norm(item_feats, axis=1, keepdims=True) + 1e-8)

    return (
        torch.tensor(cat_norm, dtype=torch.float32),
        torch.tensor(item_norm, dtype=torch.float32),
        event_onsets,
        pca,
    )
