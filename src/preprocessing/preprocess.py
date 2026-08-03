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
    # Standard 62-channel set from sub-01 to sub-09
    STANDARD_62 = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7',
                    'CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8',
                    'TP10','CP6','CP2','FCz','C4','T8','FT10','FC6','FC2',
                    'F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3',
                    'C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4',
                    'PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8',
                    'F6','AF8','AF4','F2']
    
    # Drop any channel not in the standard set
    drop_chs = [ch for ch in raw.ch_names if ch not in STANDARD_62]
    if drop_chs:
        raw.drop_channels(drop_chs)

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


def extract_trial_epochs(raw):
    """
    Extract epochs time-locked to N-back stimulus onset (code 6021).
    The label comes from the filename (zeroBACK/oneBACK/twoBACK),
    not from the event code.
    
    Returns: numpy array of shape (n_trials, n_ch, n_times)
    """
    events, event_id = mne.events_from_annotations(raw)
    
    if '6021' not in event_id:
        return None
    
    epochs = mne.Epochs(
        raw,
        events,
        event_id={'6021': event_id['6021']},
        tmin=0.0,
        tmax=2.0,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs.get_data()   # (n_trials, n_ch, n_times)


def process_file(set_path: str, label: int) -> dict:
    print(f"  Loading: {set_path.split('/')[-1]}  label={label}")

    raw      = load_and_clean(set_path)
    sfreq    = raw.info["sfreq"]
    ch_names = raw.ch_names

    events, event_id = mne.events_from_annotations(raw)

    # Stimulus code per condition — middle digit = N-back level
    # zeroBACK→6021, oneBACK→6121, twoBACK→6221, RS_Beg_EC→use 6021 if present
    STIMULUS_CODES = ['6021', '6121', '6221']
    stim_code = None
    for code in STIMULUS_CODES:
        if code in event_id:
            stim_code = code
            break

    if stim_code is None:
        raise ValueError(f"No stimulus code found in {set_path}. Available: {list(event_id.keys())}")

    epo = mne.Epochs(
        raw, events,
        event_id={stim_code: event_id[stim_code]},
        tmin=0.0, tmax=2.0,
        baseline=None,
        preload=True,
        verbose=False
    )
    trial_data = epo.get_data()   # (n_trials, n_ch, n_times)
    n_trials   = trial_data.shape[0]

    band_epochs = {}
    for band, (lo, hi) in FREQ_BANDS.items():
        # Filter continuous signal first
        raw_filt = raw.copy().filter(l_freq=lo, h_freq=hi, 
                                      fir_window='hamming', verbose=False)
        # Then extract trial epochs from filtered signal
        epo_filt = mne.Epochs(
            raw_filt, events,
            event_id={stim_code: event_id[stim_code]},
            tmin=0.0, tmax=2.0,
            baseline=None,
            preload=True,
            verbose=False
        )
        band_epochs[band] = epo_filt.get_data()   # (n_trials, n_ch, 1001)

    print(f"    → {len(ch_names)} ch  |  {n_trials} trials (code {stim_code})  |  {sfreq} Hz")

    return {
        "band_epochs": band_epochs,
        "label":       label,
        "n_epochs":    n_trials,
        "n_channels":  len(ch_names),
        "sfreq":       sfreq,
        "ch_names":    ch_names,
    }