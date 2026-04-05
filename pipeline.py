"""
pipeline.py – Audio processing and ML inference helpers for the
Percussion-Based Delamination Detection web app.

This module faithfully replicates the Homework 3 pipeline:
  1. Load & preprocess audio
  2. Segment multi-hit recordings into individual strikes
  3. Extract PSD (128 bins) + MFCC (13 mean + 13 std) = 154 features per hit
  4. Scale features with the saved StandardScaler
  5. Run inference with all four saved classifiers
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from scipy.signal import welch

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HitSegment:
    """One detected percussion strike."""
    index: int
    start_sample: int
    end_sample: int
    waveform: np.ndarray
    peak_sample: int  # absolute sample position of the peak


@dataclass
class FileResult:
    """All processing results for a single uploaded file."""
    filename: str
    sr: int
    raw_waveform: np.ndarray
    envelope: np.ndarray
    peaks: np.ndarray                       # sample indices of detected peaks
    hits: List[HitSegment] = field(default_factory=list)
    features: Optional[np.ndarray] = None   # (n_hits, 154)
    scaled_features: Optional[np.ndarray] = None
    predictions: Optional[dict] = None      # model_name -> array of 0/1
    true_label: Optional[int] = None        # from filename convention


# ---------------------------------------------------------------------------
# Config & artifact loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_artifacts(base_dir: str | Path = ".") -> Tuple[dict, object, dict]:
    """Return (config, scaler, models_dict)."""
    base = Path(base_dir)
    config = load_config(base / "config.json")
    scaler = joblib.load(base / "scaler.pkl")

    model_files = {
        "KNN": "knn_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "SVM": "svm_model.pkl",
    }
    models = {name: joblib.load(base / fname) for name, fname in model_files.items()}
    return config, scaler, models


# ---------------------------------------------------------------------------
# Label extraction from filename
# ---------------------------------------------------------------------------

def label_from_filename(filename: str) -> Optional[int]:
    """Return 0 (healthy) / 1 (unhealthy) / None based on suffix convention."""
    stem = Path(filename).stem.lower()
    if stem.endswith("_g"):
        return 0  # good / healthy
    elif stem.endswith("_b"):
        return 1  # bad / unhealthy
    return None


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(file_bytes: bytes, sr_target: int | None = None) -> Tuple[np.ndarray, int]:
    """Load WAV from bytes. Returns (signal_mono, sample_rate)."""
    import soundfile as sf
    import io

    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float64")
    # Convert to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# ---------------------------------------------------------------------------
# Segmentation (faithful to HW3 pipeline)
# ---------------------------------------------------------------------------

def compute_envelope(signal: np.ndarray, sr: int, win_sec: float) -> np.ndarray:
    """Smoothed amplitude envelope using a sliding RMS window."""
    win_len = max(int(win_sec * sr), 1)
    squared = signal ** 2
    kernel = np.ones(win_len) / win_len
    env = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return env


def detect_peaks(envelope: np.ndarray, sr: int, cfg_seg: dict) -> np.ndarray:
    """Detect peaks in the envelope matching HW3 parameters."""
    from scipy.signal import find_peaks

    min_distance = int(cfg_seg["min_peak_distance_sec"] * sr)
    threshold = envelope.mean() + cfg_seg["peak_height_factor"] * envelope.std()
    ignore_samples = int(cfg_seg["ignore_start_sec"] * sr)

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance,
    )
    # Remove peaks in the ignore zone
    peaks = peaks[peaks >= ignore_samples]
    return peaks


def extract_hits(
    signal: np.ndarray,
    sr: int,
    peaks: np.ndarray,
    cfg_seg: dict,
) -> List[HitSegment]:
    """Cut windows around each detected peak."""
    pre = int(cfg_seg["pre_hit_sec"] * sr)
    post = int(cfg_seg["post_hit_sec"] * sr)
    hits = []
    for i, pk in enumerate(peaks):
        start = max(pk - pre, 0)
        end = min(pk + post, len(signal))
        hits.append(HitSegment(
            index=i,
            start_sample=start,
            end_sample=end,
            waveform=signal[start:end],
            peak_sample=int(pk),
        ))
    return hits


# ---------------------------------------------------------------------------
# Feature extraction (PSD + MFCC, matching HW3)
# ---------------------------------------------------------------------------

def extract_psd_features(hit_wav: np.ndarray, sr: int, n_fft: int, n_bins: int) -> np.ndarray:
    """Power spectral density, resampled to n_bins."""
    nperseg = min(n_fft, len(hit_wav))
    freqs, psd = welch(hit_wav, fs=sr, nperseg=nperseg)
    # Resample PSD to fixed number of bins via interpolation
    target_freqs = np.linspace(freqs[0], freqs[-1], n_bins)
    psd_interp = np.interp(target_freqs, freqs, psd)
    return psd_interp


def extract_mfcc_features(hit_wav: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    """MFCC mean and std across time frames → 2*n_mfcc features."""
    import librosa

    # Ensure signal is long enough for at least one frame
    if len(hit_wav) < n_fft:
        hit_wav = np.pad(hit_wav, (0, n_fft - len(hit_wav)))

    mfccs = librosa.feature.mfcc(
        y=hit_wav.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])


def extract_features_for_hit(hit_wav: np.ndarray, sr: int, cfg_feat: dict) -> np.ndarray:
    """Full 154-d feature vector for one hit."""
    psd = extract_psd_features(hit_wav, sr, cfg_feat["n_fft"], cfg_feat["n_psd_bins"])
    mfcc = extract_mfcc_features(hit_wav, sr, cfg_feat["n_mfcc"], cfg_feat["n_fft"], cfg_feat["hop_length"])
    return np.concatenate([psd, mfcc])


# ---------------------------------------------------------------------------
# Full processing pipeline for one file
# ---------------------------------------------------------------------------

def process_file(
    filename: str,
    file_bytes: bytes,
    config: dict,
    scaler,
    models: dict,
) -> FileResult:
    """Run the complete pipeline on an uploaded file and return results."""
    cfg_seg = config["segmentation"]
    cfg_feat = config["feature_extraction"]

    # Load audio
    signal, sr = load_audio(file_bytes)

    # Envelope & peak detection
    envelope = compute_envelope(signal, sr, cfg_seg["envelope_win_sec"])
    peaks = detect_peaks(envelope, sr, cfg_seg)

    # Segment hits
    hits = extract_hits(signal, sr, peaks, cfg_seg)

    result = FileResult(
        filename=filename,
        sr=sr,
        raw_waveform=signal,
        envelope=envelope,
        peaks=peaks,
        hits=hits,
        true_label=label_from_filename(filename),
    )

    if len(hits) == 0:
        return result

    # Feature extraction
    feat_matrix = np.array([
        extract_features_for_hit(h.waveform, sr, cfg_feat)
        for h in hits
    ])
    result.features = feat_matrix

    # Scaling
    result.scaled_features = scaler.transform(feat_matrix)

    # Prediction
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(result.scaled_features)
    result.predictions = preds

    return result
