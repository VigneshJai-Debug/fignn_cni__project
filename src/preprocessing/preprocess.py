"""
preprocess.py
-------------
Loads one EEGLAB .set file, cleans it, band-filters into 4 bands,
epochs into 2-second windows, and returns numpy arrays ready for PLV.

Why each step exists:
  - Drop ECG1     : heart signal, not brain, throws off PLV
  - Notch 50 Hz   : Indian power-line noise
  - Avg reference : removes common-mode drift across all channels
  - Bandpass x4   : separates brain rhythms we care about
  - Epoch 2s      : gives us many short samples to compute PLV on
"""

import mne
import numpy as np

mne.set_log_level("WARNING")

FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

EPOCH_DURATION = 2.0   # seconds
EPOCH_OVERLAP  = 0.0   # seconds (no overlap for now — keeps samples independent)
PLV_THRESHOLD  = 0.3   # edges below this are pruned when building the graph


def load_and_clean(set_path: str) -> mne.io.Raw:
    """
    Load one .set file and apply basic cleaning.
    Returns a cleaned Raw object with only EEG channels.
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    # 1. Drop ECG channel — it's a heart electrode, not brain
    ecg_chs = [ch for ch in raw.ch_names if "ECG" in ch.upper()]
    if ecg_chs:
        raw.drop_channels(ecg_chs)

    # 2. Mark remaining channels as EEG type explicitly
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # 3. Notch filter at 50 Hz (India grid frequency)
    raw.notch_filter(freqs=50.0, fir_window="hamming")

    # 4. Average reference — standard for scalp EEG
    raw.set_eeg_reference("average", projection=False)

    return raw


def bandpass_filter(raw: mne.io.Raw, band: str) -> np.ndarray:
    """
    Apply a bandpass filter for the given band name.
    Returns raw signal as numpy array: (n_channels, n_times)
    """
    lo, hi = FREQ_BANDS[band]
    raw_filt = raw.copy().filter(l_freq=lo, h_freq=hi, fir_window="hamming")
    return raw_filt.get_data()   # shape: (n_ch, n_times)


def epoch_signal(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Cut a continuous signal into fixed-length non-overlapping epochs.

    Args:
        data  : (n_channels, n_times)
        sfreq : sampling frequency in Hz

    Returns:
        epochs : (n_epochs, n_channels, epoch_samples)
    """
    n_ch, n_times = data.shape
    epoch_samples = int(EPOCH_DURATION * sfreq)
    n_epochs = n_times // epoch_samples          # integer division, drops the tail

    # Slice into epochs
    epochs = np.stack(
        [data[:, i * epoch_samples : (i + 1) * epoch_samples]
         for i in range(n_epochs)],
        axis=0
    )
    return epochs  # (n_epochs, n_ch, epoch_samples)


def process_file(set_path: str, label: int) -> dict:
    """
    Full preprocessing pipeline for one .set file.

    Returns a dict with:
        'band_epochs' : { band_name: (n_epochs, n_ch, epoch_samples) }
        'label'       : int (0 / 1 / 2 for zero/one/two back)
        'n_epochs'    : int
        'n_channels'  : int
        'sfreq'       : float
        'ch_names'    : list of str
    """
    print(f"  Loading: {set_path.split('/')[-1]}  label={label}")

    raw = load_and_clean(set_path)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    band_epochs = {}
    for band in FREQ_BANDS:
        signal = bandpass_filter(raw, band)       # (n_ch, n_times)
        epochs = epoch_signal(signal, sfreq)      # (n_epochs, n_ch, epoch_samples)
        band_epochs[band] = epochs

    # All bands have the same n_epochs — use delta to count
    n_epochs = band_epochs["delta"].shape[0]

    print(f"    → {len(ch_names)} channels  |  {n_epochs} epochs  |  {sfreq} Hz")

    return {
        "band_epochs": band_epochs,
        "label":       label,
        "n_epochs":    n_epochs,
        "n_channels":  len(ch_names),
        "sfreq":       sfreq,
        "ch_names":    ch_names,
    }