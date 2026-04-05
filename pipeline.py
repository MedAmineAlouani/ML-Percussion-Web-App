from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, welch


@dataclass
class HitSegment:
    index: int
    start_sample: int
    end_sample: int
    waveform: np.ndarray
    peak_sample: int


@dataclass
class FileResult:
    filename: str
    sr: int
    raw_waveform: np.ndarray
    trimmed_waveform: np.ndarray
    envelope: np.ndarray
    peaks: np.ndarray
    threshold: float
    hits: List[HitSegment] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    scaled_features: Optional[np.ndarray] = None
    predictions: Optional[dict] = None
    true_label: Optional[int] = None


def load_config(config_path: str | Path = "config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_artifacts(base_dir: str | Path = ".") -> Tuple[dict, object, dict]:
    base = Path(base_dir)

    config = load_config(base / "config.json")
    scaler = joblib.load(base / "scaler.pkl")

    models = {
        "KNN": joblib.load(base / "knn_model.pkl"),
        "Decision Tree": joblib.load(base / "decision_tree_model.pkl"),
        "Logistic Regression": joblib.load(base / "logistic_regression_model.pkl"),
        "SVM": joblib.load(base / "svm_model.pkl"),
    }

    return config, scaler, models


def label_from_filename(filename: str) -> Optional[int]:
    file_name = Path(filename).name.lower()

    if "_g.wav" in file_name:
        return 0
    elif "_b.wav" in file_name:
        return 1
    else:
        return None


def load_audio(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float64")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data.astype(np.float64), sr


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def compute_envelope(signal_trimmed: np.ndarray, sr: int, envelope_win_sec: float) -> np.ndarray:
    envelope_win = max(1, int(envelope_win_sec * sr))
    envelope_kernel = np.ones(envelope_win) / envelope_win
    envelope = np.convolve(np.abs(signal_trimmed), envelope_kernel, mode="same")
    return envelope


def split_into_hits(signal: np.ndarray, sr: int, cfg_seg: dict):
    start_idx = int(cfg_seg["ignore_start_sec"] * sr)
    signal_trimmed = signal[start_idx:]

    envelope = compute_envelope(signal_trimmed, sr, cfg_seg["envelope_win_sec"])

    min_distance = int(cfg_seg["min_peak_distance_sec"] * sr)
    threshold = np.mean(envelope) + cfg_seg["peak_height_factor"] * np.std(envelope)

    peak_indices, _ = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance
    )

    hit_segments = []
    pre_samples = int(cfg_seg["pre_hit_sec"] * sr)
    post_samples = int(cfg_seg["post_hit_sec"] * sr)

    for i, peak_idx in enumerate(peak_indices):
        start = max(0, peak_idx - pre_samples)
        end = min(len(signal_trimmed), peak_idx + post_samples)

        hit = signal_trimmed[start:end]

        if len(hit) > 0:
            hit_segments.append(
                HitSegment(
                    index=i,
                    start_sample=start,
                    end_sample=end,
                    waveform=hit,
                    peak_sample=int(peak_idx),
                )
            )

    return signal_trimmed, envelope, peak_indices, threshold, hit_segments


def extract_psd_features(signal: np.ndarray, sr: int, n_bins: int) -> np.ndarray:
    _, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    psd = np.log10(psd + 1e-12)

    if len(psd) >= n_bins:
        psd_features = psd[:n_bins]
    else:
        psd_features = np.pad(psd, (0, n_bins - len(psd)), mode="constant")

    return psd_features


def extract_mfcc_features(signal: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
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
    psd_features = extract_psd_features(
        hit_wav,
        sr,
        cfg_feat["n_psd_bins"]
    )

    mfcc_features = extract_mfcc_features(
        hit_wav,
        sr,
        cfg_feat["n_mfcc"],
        cfg_feat["n_fft"],
        cfg_feat["hop_length"]
    )

    return np.concatenate([psd_features, mfcc_features])


def process_file(
    filename: str,
    file_bytes: bytes,
    config: dict,
    scaler,
    models: dict,
) -> FileResult:
    cfg_seg = config["segmentation"]
    cfg_feat = config["feature_extraction"]

    signal, sr = load_audio(file_bytes)
    signal = normalize_signal(signal)

    signal_trimmed, envelope, peaks, threshold, hits = split_into_hits(signal, sr, cfg_seg)

    result = FileResult(
        filename=filename,
        sr=sr,
        raw_waveform=signal,
        trimmed_waveform=signal_trimmed,
        envelope=envelope,
        peaks=peaks,
        threshold=threshold,
        hits=hits,
        true_label=label_from_filename(filename),
    )

    if len(hits) == 0:
        return result

    feat_matrix = np.array([
        extract_features_for_hit(hit.waveform, sr, cfg_feat)
        for hit in hits
    ])

    result.features = feat_matrix
    result.scaled_features = scaler.transform(feat_matrix)

    preds = {}
    for model_name, model in models.items():
        preds[model_name] = model.predict(result.scaled_features)

    result.predictions = preds

    return result
