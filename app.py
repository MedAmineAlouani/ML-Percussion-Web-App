from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from pipeline import FileResult, load_artifacts, process_file


REPO_DIR = Path(__file__).resolve().parent
LABEL_MAP = {0: "Healthy", 1: "Unhealthy"}
MODEL_ORDER = ["KNN", "Decision Tree", "Logistic Regression", "SVM"]

st.set_page_config(
    page_title="Delamination Detector",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    .stDataFrame {border-radius: 8px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)


def get_artifacts():
    return load_artifacts(REPO_DIR)


config, scaler, models = get_artifacts()

if "results" not in st.session_state:
    st.session_state.results = []


def all_hits():
    out = []
    for r in st.session_state.results:
        for h in r.hits:
            out.append((r, h))
    return out


def plot_waveform_and_envelope(result: FileResult):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=110, sharex=False)
    fig.patch.set_facecolor("white")

    t_raw = np.arange(len(result.raw_waveform)) / result.sr
    axes[0].plot(t_raw, result.raw_waveform, color="#4a6fa5", linewidth=0.4, alpha=0.85)
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].set_title("Raw Waveform", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.25)

    t_trim = np.arange(len(result.trimmed_waveform)) / result.sr
    axes[1].plot(t_trim, result.trimmed_waveform, color="#4a6fa5", linewidth=0.35, alpha=0.45, label="Trimmed signal")
    axes[1].plot(t_trim, result.envelope, color="#e07a3a", linewidth=1.0, label="Envelope")

    axes[1].axhline(result.threshold, color="#d62828", linestyle="--", linewidth=1.0, label="Threshold")

    if len(result.peaks) > 0:
        axes[1].plot(
            result.peaks / result.sr,
            result.envelope[result.peaks],
            "v",
            color="#d62828",
            markersize=7,
            label=f"Peaks ({len(result.peaks)})",
        )

    for h in result.hits:
        t_start = h.start_sample / result.sr
        t_end = h.end_sample / result.sr
        axes[1].axvspan(t_start, t_end, alpha=0.12, color="#2a9d8f")

    axes[1].set_ylabel("Amplitude / Envelope", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].set_title("Trimmed Signal, Envelope, and Detected Peaks", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def plot_hit_waveform(hit, sr):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=110)
    fig.patch.set_facecolor("white")
    t = np.arange(len(hit.waveform)) / sr
    ax.plot(t, hit.waveform, color="#4a6fa5", linewidth=0.6)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_title(f"Hit {hit.index + 1} Waveform", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_psd(psd_vec, sr):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=110)
    fig.patch.set_facecolor("white")
    freqs = np.linspace(0, sr / 2, len(psd_vec))
    ax.plot(freqs, psd_vec, color="#6a4c93", linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Log PSD", fontsize=10)
    ax.set_title("Power Spectral Density", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_mfcc(mfcc_mean, mfcc_std, n_mfcc):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=110)
    fig.patch.set_facecolor("white")
    x = np.arange(n_mfcc)
    ax.bar(x, mfcc_mean, yerr=mfcc_std, capsize=3, color="#2a9d8f", alpha=0.85)
    ax.set_xlabel("MFCC Coefficient", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("MFCC Mean ± Std", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(3.5, 3), dpi=110)
    fig.patch.set_facecolor("white")
    ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Unhealthy"], fontsize=9)
    ax.set_yticklabels(["Healthy", "Unhealthy"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(model_name, fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    fig.tight_layout()
    return fig


tabs = st.tabs([
    "🏠 Home",
    "📤 Upload & Process",
    "📊 Segmentation Viewer",
    "🔬 Feature Viewer",
    "🤖 Predictions",
    "📈 Evaluation",
    "ℹ️ About",
])


with tabs[0]:
    st.title("Percussion-Based Delamination Detection")
    st.markdown("""
    This app follows the final Homework 3 notebook pipeline as closely as possible.

    Workflow:
    1. Upload `.wav` file(s)
    2. Normalize audio
    3. Ignore first 0.15 s for hit detection
    4. Build moving-average envelope of `abs(signal)`
    5. Detect peaks
    6. Extract fixed hit windows
    7. Extract PSD + MFCC features
    8. Scale features with saved scaler
    9. Predict with 4 saved models
    """)

    st.subheader("Filename Convention")
    st.markdown("""
    - `*_g.wav` → Healthy
    - `*_b.wav` → Unhealthy
    """)


with tabs[1]:
    st.header("Upload & Process")

    uploaded = st.file_uploader(
        "Choose WAV file(s)",
        type=["wav"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("🚀 Process uploaded files", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Processing ...")

            for idx, f in enumerate(uploaded):
                progress.progress(
                    idx / len(uploaded),
                    text=f"Processing {f.name} ({idx + 1}/{len(uploaded)}) ..."
                )

                try:
                    r = process_file(f.name, f.read(), config, scaler, models)
                    results.append(r)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")

            progress.progress(1.0, text="Done!")
            st.session_state.results = results

            total_hits = sum(len(r.hits) for r in results)
            labeled = sum(1 for r in results if r.true_label is not None)

            c1, c2, c3 = st.columns(3)
            c1.metric("Files processed", len(results))
            c2.metric("Total hits detected", total_hits)
            c3.metric("Files with labels", labeled)

            for r in results:
                if len(r.hits) > 0:
                    st.success(f"{r.filename} — {len(r.hits)} hit(s) detected")
                else:
                    st.warning(f"{r.filename} — No hits detected")


with tabs[2]:
    st.header("Segmentation Viewer")

    if not st.session_state.results:
        st.info("Process files first.")
    else:
        file_names = [r.filename for r in st.session_state.results]
        sel = st.selectbox("Select a file", file_names, key="seg_file")
        result = st.session_state.results[file_names.index(sel)]

        st.write("Sample rate:", result.sr)
        st.write("Detected hits:", len(result.hits))
        st.write("Threshold:", float(result.threshold))

        fig = plot_waveform_and_envelope(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if result.features is not None:
            st.write("Feature matrix shape:", result.features.shape)
            st.write("Scaled feature matrix shape:", result.scaled_features.shape)

        st.markdown("### Current parameters")
        st.json(config)

        if result.hits:
            st.markdown("### Individual hit previews")
            cols = st.columns(min(4, len(result.hits)))
            for i, h in enumerate(result.hits):
                with cols[i % len(cols)]:
                    fig2 = plot_hit_waveform(h, result.sr)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)


with tabs[3]:
    st.header("Feature Viewer")

    if not st.session_state.results:
        st.info("Process files first.")
    else:
        hit_list = all_hits()

        if not hit_list:
            st.warning("No hits detected.")
        else:
            options = [f"{r.filename} — Hit {h.index + 1}" for r, h in hit_list]
            sel_idx = st.selectbox("Select a hit", range(len(options)), format_func=lambda i: options[i])

            r, h = hit_list[sel_idx]
            cfg_feat = config["feature_extraction"]

            feat = r.features[h.index]
            n_psd = cfg_feat["n_psd_bins"]
            n_mfcc = cfg_feat["n_mfcc"]

            psd_vec = feat[:n_psd]
            mfcc_mean = feat[n_psd:n_psd + n_mfcc]
            mfcc_std = feat[n_psd + n_mfcc:]

            st.metric("Feature vector length", len(feat))

            c1, c2 = st.columns(2)

            with c1:
                fig_psd = plot_psd(psd_vec, r.sr)
                st.pyplot(fig_psd, use_container_width=True)
                plt.close(fig_psd)

            with c2:
                fig_mfcc = plot_mfcc(mfcc_mean, mfcc_std, n_mfcc)
                st.pyplot(fig_mfcc, use_container_width=True)
                plt.close(fig_mfcc)


with tabs[4]:
    st.header("Predictions")

    if not st.session_state.results:
        st.info("Process files first.")
    else:
        all_rows = []

        for r in st.session_state.results:
            if r.predictions is None:
                continue

            for h in r.hits:
                row = {"File": r.filename, "Hit": h.index + 1}
                for m in MODEL_ORDER:
                    pred = int(r.predictions[m][h.index])
                    row[m] = LABEL_MAP[pred]

                votes = [r.predictions[m][h.index] for m in MODEL_ORDER]
                majority = int(np.round(np.mean(votes)))
                row["Majority Vote"] = LABEL_MAP[majority]

                if r.true_label is not None:
                    row["True Label"] = LABEL_MAP[r.true_label]

                all_rows.append(row)

        if all_rows:
            df = pd.DataFrame(all_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("## File-Level Summary")

            for r in st.session_state.results:
                if r.predictions is None or len(r.hits) == 0:
                    continue

                st.markdown(f"### {r.filename}")

                cols = st.columns(len(MODEL_ORDER) + 1)

                for i, m in enumerate(MODEL_ORDER):
                    pct_unhealthy = r.predictions[m].mean() * 100
                    cols[i].metric(m, f"{pct_unhealthy:.0f}% unhealthy")

                all_votes = np.array([r.predictions[m] for m in MODEL_ORDER])
                majority_per_hit = np.round(all_votes.mean(axis=0)).astype(int)
                pct = majority_per_hit.mean() * 100
                cols[-1].metric("Majority Vote", f"{pct:.0f}% unhealthy")

            csv = df.to_csv(index=False)
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")


with tabs[5]:
    st.header("Evaluation")

    labeled_results = [
        r for r in st.session_state.results
        if r.true_label is not None and r.predictions is not None and len(r.hits) > 0
    ]

    if not st.session_state.results:
        st.info("Process files first.")
    elif not labeled_results:
        st.warning("No labeled files available for evaluation.")
    else:
        y_true_all = []
        preds_all = {m: [] for m in MODEL_ORDER}

        for r in labeled_results:
            for h in r.hits:
                y_true_all.append(r.true_label)
                for m in MODEL_ORDER:
                    preds_all[m].append(int(r.predictions[m][h.index]))

        y_true_all = np.array(y_true_all)

        st.write(f"Evaluating on {len(y_true_all)} hits")

        cols = st.columns(len(MODEL_ORDER))
        for i, m in enumerate(MODEL_ORDER):
            acc = accuracy_score(y_true_all, preds_all[m]) * 100
            cols[i].metric(m, f"{acc:.1f}%")

        st.markdown("## Confusion Matrices")
        cols = st.columns(len(MODEL_ORDER))
        for i, m in enumerate(MODEL_ORDER):
            with cols[i]:
                fig_cm = plot_confusion_matrix(y_true_all, preds_all[m], m)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

        rows = []
        for m in MODEL_ORDER:
            p, r_val, f1, _ = precision_recall_fscore_support(
                y_true_all,
                preds_all[m],
                labels=[0, 1],
                average="binary",
                pos_label=1,
                zero_division=0
            )
            acc = accuracy_score(y_true_all, preds_all[m])

            rows.append({
                "Model": m,
                "Accuracy": f"{acc:.3f}",
                "Precision": f"{p:.3f}",
                "Recall": f"{r_val:.3f}",
                "F1-Score": f"{f1:.3f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tabs[6]:
    st.header("About This App")

    st.markdown("""
    This app uses the saved Homework 3 models and the same notebook-style pipeline.

    ### Final model results from the homework
    - SVM: 80.42% test accuracy
    - Logistic Regression: 74.87% test accuracy
    - KNN: 65.08% test accuracy
    - Decision Tree: 62.96% test accuracy

    ### Important segmentation settings
    - ignore_start_sec = 0.15
    - envelope_win_sec = 0.01
    - min_peak_distance_sec = 0.30
    - peak_height_factor = 2.5
    - pre_hit_sec = 0.016
    - post_hit_sec = 0.08

    ### Important feature settings
    - n_mfcc = 13
    - n_fft = 2048
    - hop_length = 512
    - n_psd_bins = 128

    ### Feature vector
    - PSD = 128 values
    - MFCC mean/std = 26 values
    - Total = 154 features
    """)

    with st.expander("Show loaded config"):
        st.json(config)
