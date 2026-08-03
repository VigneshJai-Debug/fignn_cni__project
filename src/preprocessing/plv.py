"""
plv.py
------
Computes Phase Locking Value (PLV) matrices from epoched EEG.

What PLV is:
  For two channels i and j, PLV measures how consistently their
  instantaneous phases stay locked together over the epoch window.
  PLV = 1 → always in sync. PLV = 0 → completely random phase relation.

How we get instantaneous phase:
  Hilbert transform converts a real signal x(t) into a complex
  analytic signal. The angle of that complex number at each time
  point is the instantaneous phase φ(t).

Why 4 bands separately:
  Synchrony in theta (4-8 Hz) means something different from
  synchrony in beta (13-30 Hz). Keeping bands separate lets the
  GAT + band-weight layer learn which band matters most for load.
"""

import numpy as np
from scipy.signal import hilbert


def compute_plv_matrix(epoch: np.ndarray) -> np.ndarray:
    """
    Compute PLV between every pair of channels for one epoch.

    Args:
        epoch : (n_channels, n_timepoints)  — already bandpass filtered

    Returns:
        plv   : (n_channels, n_channels)    — symmetric, values in [0, 1]
    """
    n_ch = epoch.shape[0]

    # Hilbert transform → analytic signal → extract phase
    analytic = hilbert(epoch, axis=1)          # (n_ch, n_times) complex
    phase    = np.angle(analytic)              # (n_ch, n_times) real, in [-π, π]

    plv = np.zeros((n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            # Phase difference at every timepoint
            delta_phi = phase[i] - phase[j]

            # PLV = magnitude of mean unit complex exponential
            # If phases consistently differ by same amount → |mean| close to 1
            # If phases differ randomly → vectors cancel → |mean| close to 0
            plv_ij = np.abs(np.mean(np.exp(1j * delta_phi)))

            plv[i, j] = plv_ij
            plv[j, i] = plv_ij   # symmetric

    np.fill_diagonal(plv, 1.0)   # channel perfectly in sync with itself
    return plv


def compute_all_band_plv(band_epoch_dict: dict, epoch_idx: int) -> dict:
    """
    Compute PLV matrices for one epoch across all 4 bands.

    Args:
        band_epoch_dict : { band: (n_epochs, n_ch, n_times) }
        epoch_idx       : which epoch to compute PLV for

    Returns:
        { band: (n_ch, n_ch) PLV matrix }
    """
    return {
        band: compute_plv_matrix(band_epoch_dict[band][epoch_idx])
        for band in band_epoch_dict
    }


def batch_compute_plv(band_epoch_dict: dict, verbose: bool = True) -> list:
    """
    Compute PLV + correlation for every epoch across all bands.
    Returns list of dicts with 'plv' and 'corr' keys.
    """
    n_epochs = band_epoch_dict["delta"].shape[0]
    connectivity_list = []

    for i in range(n_epochs):
        if verbose and i % 20 == 0:
            print(f"    Connectivity: epoch {i}/{n_epochs}")
        connectivity_list.append(
            compute_all_band_connectivity(band_epoch_dict, i)
        )
    return connectivity_list

def compute_correlation_matrix(epoch: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between every pair of channels.
    Captures amplitude correlation — complementary to PLV phase sync.
    
    Args:
        epoch : (n_channels, n_timepoints)
    Returns:
        corr  : (n_channels, n_channels) symmetric, values in [-1, 1]
    """
    # Z-score each channel
    mean = epoch.mean(axis=1, keepdims=True)
    std  = epoch.std(axis=1, keepdims=True) + 1e-30
    z    = (epoch - mean) / std          # (n_ch, n_times)
    
    # Correlation matrix
    corr = (z @ z.T) / epoch.shape[1]   # (n_ch, n_ch)
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr.astype(np.float32)


def compute_all_band_connectivity(band_epoch_dict: dict, epoch_idx: int) -> dict:
    """
    Compute both PLV and correlation for one epoch across all bands.
    Returns dict with 'plv' and 'corr' keys, each containing band dicts.
    """
    result = {'plv': {}, 'corr': {}}
    for band in band_epoch_dict:
        epoch = band_epoch_dict[band][epoch_idx]
        result['plv'][band]  = compute_plv_matrix(epoch)
        result['corr'][band] = compute_correlation_matrix(epoch)
    return result