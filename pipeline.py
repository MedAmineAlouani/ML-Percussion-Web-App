"""
pipeline.py – Audio processing and ML inference helpers for the
Percussion-Based Delamination Detection web app.

This version is aligned with the final Homework 3 notebook pipeline:
  1. Load audio
  2. Convert to mono if needed
  3. Normalize signal
  4. Ignore the first 0.15 s during peak detection
  5. Build envelope from moving average of abs(signal)
  6. Detect peaks using mean(envelope) + 2.5 * std(envelope)
  7. Extract fixed windows around each detected peak
  8. Extract PSD + MFCC features
  9. Scale with saved StandardScaler
 10. Run inference with saved classifiers
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, welch


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HitSegment:
    index: int
    start_sample: int
    end_sample: int
    waveform: np.ndarray
    peak_sample: int  # peak index in the trimmed signal


@dataclass
class FileResult:
    filename: str
    sr: int
    raw_waveform: np.ndarray
    trimmed_waveform: np.ndarray
    envelope: np.ndarray
    peaks: np.ndarray
    hits: List[HitSegment] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    scaled_features: Optional[np.ndarray] = None
    predictions: Optional[dict] = None
    true_label: Optional[int] = None
    threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Config & artifact loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_artifacts(base_dir: str | Path = ".") -> Tuple[dict, object, dict]:
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
# Labels
# ---------------------------------------------------------------------------

def label_from_filename(filename: str) -> Optional[int]:
    stem = Path(filename).stem.lower()
    if stem.endswith("_g"):
        return 0
    if stem.endswith("_b"):
        return 1
    return None


# ---------------------------------------------------------------------------
# Audio loading / normalization
# ---------------------------------------------------------------------------

def load_audio(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Load WAV from bytes and convert to mono if needed.
    Keep the original sample rate.
    """
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float64")

    if data.ndim > 1:
        data = data.mean(axis=1)

    return data.astype(np.float64), sr


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Match the notebook normalization:
    divide by max absolute value if nonzero.
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def compute_envelope(signal_trimmed: np.ndarray, sr: int, win_sec: float) -> np.ndarray:
    """
    Match the notebook:
    envelope = moving average of abs(signal_trimmed)
    """
    win_len = max(1, int(win_sec * sr))
    kernel = np.ones(win_len) / win_len
    envelope = np.convolve(np.abs(signal_trimmed), kernel, mode="same")
    return envelope


def detect_peaks_from_envelope(envelope: np.ndarray, sr: int, cfg_seg: dict) -> Tuple[np.ndarray, float]:
    """
    Match the notebook:
    - threshold = mean(envelope) + peak_height_factor * std(envelope)
    - distance = min_peak_distance_sec * sr
    """
    min_distance = int(cfg_seg["min_peak_distance_sec"] * sr)
    threshold = np.mean(envelope) + cfg_seg["peak_height_factor"] * np.std(envelope)

    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )
    return peaks, threshold


def extract_hits(signal_trimmed: np.ndarray, sr: int, peaks: np.ndarray, cfg_seg: dict) -> List[HitSegment]:
    """
    Extract fixed windows around each detected peak using the trimmed signal.
    """
    pre_samples = int(cfg_seg["pre_hit_sec"] * sr)
    post_samples = int(cfg_seg["post_hit_sec"] * sr)

    hits = []
    for i, peak_idx in enumerate(peaks):
        start = max(0, peak_idx - pre_samples)
        end = min(len(signal_trimmed), peak_idx + post_samples)

        hit = signal_trimmed[start:end]

        if len(hit) > 0:
            hits.append(
                HitSegment(
                    index=i,
                    start_sample=start,
                    end_sample=end,
                    waveform=hit,
                    peak_sample=int(peak_idx),
                )
            )

    return hits


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_psd_features(signal: np.ndarray, sr: int, n_bins: int) -> np.ndarray:
    """
    Match the notebook:
    - Welch PSD
    - log10 scaling
    - truncate or zero-pad to fixed length
    """
    freqs, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    psd = np.log10(psd + 1e-12)

    if len(psd) >= n_bins:
        psd_features = psd[:n_bins]
    else:
        psd_features = np.pad(psd, (0, n_bins - len(psd)), mode="constant")

    return psd_features


def extract_mfcc_features(signal: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    """
    Match the notebook:
    - n_fft = min(config_n_fft, len(signal))
    - hop_length = min(config_hop_length, max(1, len(signal)//2))
    - MFCC summarized with mean and std across time
    """
    import librosa

    effective_n_fft = min(n_fft, len(signal))
    effective_hop = min(hop_length, max(1, len(signal) // 2))

    mfcc = librosa.feature.mfcc(
        y=signal.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=effective_n_fft,
        hop_length=effective_hop
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])


def extract_features_for_hit(hit_wav: np.ndarray, sr: int, cfg_feat: dict) -> np.ndarray:
    psd = extract_psd_features(
        hit_wav,
        sr,
        cfg_feat["n_psd_bins"]
    )

    mfcc = extract_mfcc_features(
        hit_wav,
        sr,
        cfg_feat["n_mfcc"],
        cfg_feat["n_fft"],
        cfg_feat["hop_length"]
    )

    return np.concatenate([psd, mfcc])


# ---------------------------------------------------------------------------
# Full file pipeline
# ---------------------------------------------------------------------------

def process_file(
    filename: str,
    file_bytes: bytes,
    config: dict,
    scaler,
    models: dict,
) -> FileResult:
    cfg_seg = config["segmentation"]
    cfg_feat = config["feature_extraction"]

    # Load and normalize audio
    signal, sr = load_audio(file_bytes)
    signal = normalize_signal(signal)

    # Match notebook behavior:
    # ignore first 0.15 s BEFORE envelope/peak detection
    start_idx = int(cfg_seg["ignore_start_sec"] * sr)
    signal_trimmed = signal[start_idx:]

    # Envelope and peak detection on trimmed signal only
    envelope = compute_envelope(signal_trimmed, sr, cfg_seg["envelope_win_sec"])
    peaks, threshold = detect_peaks_from_envelope(envelope, sr, cfg_seg)

    # Extract hit windows from trimmed signal
    hits = extract_hits(signal_trimmed, sr, peaks, cfg_seg)

    result = FileResult(
        filename=filename,
        sr=sr,
        raw_waveform=signal,
        trimmed_waveform=signal_trimmed,
        envelope=envelope,
        peaks=peaks,
        hits=hits,
        true_label=label_from_filename(filename),
        threshold=threshold,
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
