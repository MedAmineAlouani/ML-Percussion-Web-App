"""
Percussion-Based Delamination Detection – Streamlit Web App
============================================================
Upload percussion recordings, segment hits, extract features,
and classify each hit as healthy or unhealthy using four saved
ML models from the Homework 3 pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from pipeline import (
    FileResult,
    load_artifacts,
    process_file,
)

# ---------------------------------------------------------------------------
# App-wide config
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent
LABEL_MAP = {0: "Healthy", 1: "Unhealthy"}
MODEL_ORDER = ["KNN", "Decision Tree", "Logistic Regression", "SVM"]

st.set_page_config(
    page_title="Delamination Detector",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Card-like containers */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    /* Nicer tables */
    .stDataFrame {border-radius: 8px; overflow: hidden;}
    /* Status badges */
    .badge-healthy {
        background-color: #d4edda; color: #155724;
        padding: 4px 12px; border-radius: 16px; font-weight: 600;
    }
    .badge-unhealthy {
        background-color: #f8d7da; color: #721c24;
        padding: 4px 12px; border-radius: 16px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models and scaler …")
def get_artifacts():
    return load_artifacts(REPO_DIR)


config, scaler, models = get_artifacts()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results: list[FileResult] = []


def all_hits():
    """Flatten all hits across all processed files."""
    out = []
    for r in st.session_state.results:
        for h in r.hits:
            out.append((r, h))
    return out


# ---------------------------------------------------------------------------
# Plotting helpers (publication-quality Matplotlib)
# ---------------------------------------------------------------------------

def _fig(figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor("white")
    return fig, ax


def plot_waveform_and_envelope(result: FileResult):
    """Raw waveform + envelope + detected peaks."""
    t = np.arange(len(result.raw_waveform)) / result.sr

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=110, sharex=True)
    fig.patch.set_facecolor("white")

    # Waveform
    axes[0].plot(t, result.raw_waveform, color="#4a6fa5", linewidth=0.4, alpha=0.85)
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].set_title("Raw Waveform", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.25)

    # Envelope + peaks
    axes[1].plot(t, result.envelope, color="#e07a3a", linewidth=0.8, label="Envelope")
    if len(result.peaks) > 0:
        axes[1].plot(
            result.peaks / result.sr,
            result.envelope[result.peaks],
            "v", color="#d62828", markersize=8, label=f"Peaks ({len(result.peaks)})",
        )
    # Mark hit windows
    cfg_seg = config["segmentation"]
    pre = cfg_seg["pre_hit_sec"]
    post = cfg_seg["post_hit_sec"]
    for h in result.hits:
        t_start = h.start_sample / result.sr
        t_end = h.end_sample / result.sr
        axes[1].axvspan(t_start, t_end, alpha=0.12, color="#2a9d8f")

    axes[1].set_ylabel("Envelope", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].set_title("Envelope & Detected Peaks", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def plot_hit_waveform(hit, sr):
    fig, ax = _fig()
    t = np.arange(len(hit.waveform)) / sr * 1000  # ms
    ax.plot(t, hit.waveform, color="#4a6fa5", linewidth=0.6)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_title(f"Hit {hit.index + 1} Waveform", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_psd(psd_vec, sr, n_fft):
    fig, ax = _fig()
    freqs = np.linspace(0, sr / 2, len(psd_vec))
    ax.semilogy(freqs, psd_vec, color="#6a4c93", linewidth=1.0)
    ax.fill_between(freqs, psd_vec, alpha=0.15, color="#6a4c93")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("PSD", fontsize=10)
    ax.set_title("Power Spectral Density", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def plot_mfcc(mfcc_mean, mfcc_std, n_mfcc):
    fig, ax = _fig()
    x = np.arange(n_mfcc)
    ax.bar(x, mfcc_mean, yerr=mfcc_std, capsize=3,
           color="#2a9d8f", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("MFCC Coefficient", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("MFCC Mean ± Std", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.5, 3), dpi=110)
    fig.patch.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Unhealthy"], fontsize=9)
    ax.set_yticklabels(["Healthy", "Unhealthy"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(model_name, fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


# ===================================================================
#  TABS
# ===================================================================
tabs = st.tabs([
    "🏠 Home",
    "📤 Upload & Process",
    "📊 Segmentation Viewer",
    "🔬 Feature Viewer",
    "🤖 Predictions",
    "📈 Evaluation",
    "ℹ️ About",
])

# -------------------------------------------------------------------
# TAB 0 – Home
# -------------------------------------------------------------------
with tabs[0]:
    st.title("Percussion-Based Delamination Detection")
    st.markdown("""
    Welcome! This app uses **machine learning** to classify percussion recordings
    from a composite plate as **Healthy** or **Unhealthy** (delaminated).

    The idea is simple: when you tap a composite panel, the *sound* changes
    depending on whether the internal structure is intact or damaged.
    This app automates the analysis pipeline from recording to classification.
    """)

    st.markdown("---")
    st.subheader("How It Works")

    cols = st.columns(6)
    steps = [
        ("📤", "Upload", "Upload .wav files"),
        ("✂️", "Segment", "Detect individual hits"),
        ("📊", "Features", "Extract PSD & MFCC"),
        ("⚖️", "Scale", "Apply saved scaler"),
        ("🤖", "Predict", "Run 4 ML models"),
        ("✅", "Results", "View predictions"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="text-align:center; padding:12px; background:#f8f9fa;
                    border-radius:12px; border:1px solid #dee2e6;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:700; margin:4px 0;">{title}</div>
            <div style="font-size:0.85rem; color:#6c757d;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Filename Convention")
    st.markdown("""
    | Suffix | Meaning |
    |--------|---------|
    | `*_g.wav` | **Good** – healthy / intact |
    | `*_b.wav` | **Bad** – unhealthy / delaminated |

    If filenames follow this convention, the **Evaluation** tab will
    automatically compute accuracy, confusion matrices, and more.
    """)

    st.info("Navigate to the **Upload & Process** tab to get started.", icon="👉")

# -------------------------------------------------------------------
# TAB 1 – Upload & Process
# -------------------------------------------------------------------
with tabs[1]:
    st.header("Upload & Process")
    st.markdown("Upload one or more `.wav` percussion recordings to begin analysis.")

    uploaded = st.file_uploader(
        "Choose WAV file(s)",
        type=["wav"],
        accept_multiple_files=True,
        help="Supports single-hit or multi-hit recordings.",
    )

    if uploaded:
        if st.button("🚀 Process uploaded files", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Processing …")
            for idx, f in enumerate(uploaded):
                progress.progress(
                    (idx) / len(uploaded),
                    text=f"Processing **{f.name}** ({idx + 1}/{len(uploaded)}) …",
                )
                try:
                    r = process_file(f.name, f.read(), config, scaler, models)
                    results.append(r)
                except Exception as e:
                    st.error(f"Error processing **{f.name}**: {e}")

            progress.progress(1.0, text="Done!")
            st.session_state.results = results

            # Summary
            st.markdown("---")
            st.subheader("Processing Summary")
            summary_cols = st.columns(3)
            total_hits = sum(len(r.hits) for r in results)
            labeled = sum(1 for r in results if r.true_label is not None)
            summary_cols[0].metric("Files processed", len(results))
            summary_cols[1].metric("Total hits detected", total_hits)
            summary_cols[2].metric("Files with labels", labeled)

            for r in results:
                label_badge = ""
                if r.true_label is not None:
                    lbl = LABEL_MAP[r.true_label]
                    cls = "badge-healthy" if r.true_label == 0 else "badge-unhealthy"
                    label_badge = f' <span class="{cls}">{lbl}</span>'

                if len(r.hits) > 0:
                    st.success(
                        f"**{r.filename}** — {len(r.hits)} hit(s) detected, "
                        f"sample rate {r.sr} Hz"
                    )
                else:
                    st.warning(
                        f"**{r.filename}** — No hits detected. "
                        "The recording may be too quiet or lack distinct percussion strikes."
                    )
    else:
        st.info("Upload `.wav` files above, then press **Process**.", icon="📎")

# -------------------------------------------------------------------
# TAB 2 – Segmentation Viewer
# -------------------------------------------------------------------
with tabs[2]:
    st.header("Segmentation Viewer")

    if not st.session_state.results:
        st.info("Process files in the **Upload & Process** tab first.", icon="⏳")
    else:
        file_names = [r.filename for r in st.session_state.results]
        sel = st.selectbox("Select a file", file_names, key="seg_file")
        result = st.session_state.results[file_names.index(sel)]

        st.markdown(f"**Sample rate:** {result.sr} Hz &nbsp;|&nbsp; "
                    f"**Duration:** {len(result.raw_waveform)/result.sr:.2f} s &nbsp;|&nbsp; "
                    f"**Hits detected:** {len(result.hits)}")

        fig = plot_waveform_and_envelope(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if result.hits:
            st.markdown("#### Individual Hit Windows")
            hit_cols = st.columns(min(len(result.hits), 4))
            for i, h in enumerate(result.hits):
                with hit_cols[i % len(hit_cols)]:
                    st.caption(f"Hit {h.index + 1}")
                    fig2 = plot_hit_waveform(h, result.sr)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
        else:
            st.warning("No hits were detected in this file.")

# -------------------------------------------------------------------
# TAB 3 – Feature Viewer
# -------------------------------------------------------------------
with tabs[3]:
    st.header("Feature Viewer")

    if not st.session_state.results:
        st.info("Process files in the **Upload & Process** tab first.", icon="⏳")
    else:
        hit_list = all_hits()
        if not hit_list:
            st.warning("No hits detected in any uploaded file.")
        else:
            options = [f"{r.filename} — Hit {h.index + 1}" for r, h in hit_list]
            sel_idx = st.selectbox("Select a hit", range(len(options)),
                                   format_func=lambda i: options[i], key="feat_hit")
            r, h = hit_list[sel_idx]
            cfg_feat = config["feature_extraction"]
            n_psd = cfg_feat["n_psd_bins"]
            n_mfcc = cfg_feat["n_mfcc"]

            feat = r.features[h.index]  # 154-d
            psd_vec = feat[:n_psd]
            mfcc_mean = feat[n_psd:n_psd + n_mfcc]
            mfcc_std = feat[n_psd + n_mfcc:]

            st.metric("Feature vector length", f"{len(feat)} dimensions")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Power Spectral Density (PSD)")
                fig_psd = plot_psd(psd_vec, r.sr, cfg_feat["n_fft"])
                st.pyplot(fig_psd, use_container_width=True)
                plt.close(fig_psd)
                st.caption(
                    "PSD captures how the signal's power is distributed across "
                    "frequencies. Delamination typically shifts energy to lower "
                    "frequencies."
                )
            with col_b:
                st.markdown("#### Mel-Frequency Cepstral Coefficients (MFCC)")
                fig_mfcc = plot_mfcc(mfcc_mean, mfcc_std, n_mfcc)
                st.pyplot(fig_mfcc, use_container_width=True)
                plt.close(fig_mfcc)
                st.caption(
                    "MFCCs summarize the spectral shape on a perceptual (mel) "
                    "scale. Mean and standard deviation across time frames are "
                    "used as features."
                )

# -------------------------------------------------------------------
# TAB 4 – Predictions
# -------------------------------------------------------------------
with tabs[4]:
    st.header("Predictions")

    if not st.session_state.results:
        st.info("Process files in the **Upload & Process** tab first.", icon="⏳")
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
                # Majority vote
                votes = [r.predictions[m][h.index] for m in MODEL_ORDER]
                majority = int(np.round(np.mean(votes)))
                row["Majority Vote"] = LABEL_MAP[majority]
                if r.true_label is not None:
                    row["True Label"] = LABEL_MAP[r.true_label]
                all_rows.append(row)

        if not all_rows:
            st.warning("No hits to show predictions for.")
        else:
            df = pd.DataFrame(all_rows)

            # ---- Hit-level table ----
            st.subheader("Hit-Level Predictions")

            def color_pred(val):
                if val == "Healthy":
                    return "background-color: #d4edda; color: #155724"
                elif val == "Unhealthy":
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            styled = df.style.map(
                color_pred,
                subset=[c for c in df.columns if c not in ("File", "Hit")],
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ---- File-level summary ----
            st.subheader("File-Level Summary")
            for r in st.session_state.results:
                if r.predictions is None or len(r.hits) == 0:
                    continue
                st.markdown(f"##### {r.filename}")
                n = len(r.hits)

                mcols = st.columns(len(MODEL_ORDER) + 1)
                for i, m in enumerate(MODEL_ORDER):
                    pct_unhealthy = r.predictions[m].mean() * 100
                    mcols[i].metric(
                        m,
                        f"{pct_unhealthy:.0f}% unhealthy",
                        help=f"{int(r.predictions[m].sum())}/{n} hits classified as unhealthy",
                    )
                # Majority
                all_votes = np.array([r.predictions[m] for m in MODEL_ORDER])
                majority_per_hit = np.round(all_votes.mean(axis=0)).astype(int)
                pct = majority_per_hit.mean() * 100
                mcols[-1].metric(
                    "Majority Vote",
                    f"{pct:.0f}% unhealthy",
                    help=f"{int(majority_per_hit.sum())}/{n} hits",
                )
                st.markdown("---")

            # ---- CSV download ----
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download predictions as CSV",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )

# -------------------------------------------------------------------
# TAB 5 – Evaluation
# -------------------------------------------------------------------
with tabs[5]:
    st.header("Evaluation")

    results_with_labels = [
        r for r in st.session_state.results
        if r.true_label is not None and r.predictions is not None and len(r.hits) > 0
    ]

    if not st.session_state.results:
        st.info("Process files in the **Upload & Process** tab first.", icon="⏳")
    elif not results_with_labels:
        st.warning(
            "No files have ground-truth labels. "
            "Use the filename convention `*_g.wav` (healthy) / `*_b.wav` (unhealthy) "
            "to enable automatic evaluation."
        )
    else:
        # Collect ground truth and predictions at hit level
        y_true_all = []
        preds_all = {m: [] for m in MODEL_ORDER}
        for r in results_with_labels:
            for h in r.hits:
                y_true_all.append(r.true_label)
                for m in MODEL_ORDER:
                    preds_all[m].append(int(r.predictions[m][h.index]))

        y_true_all = np.array(y_true_all)

        st.markdown(f"**Evaluating on {len(y_true_all)} hits** from "
                    f"{len(results_with_labels)} labeled file(s).")

        # ---- Accuracy cards ----
        st.subheader("Accuracy")
        acc_cols = st.columns(len(MODEL_ORDER))
        for i, m in enumerate(MODEL_ORDER):
            acc = accuracy_score(y_true_all, preds_all[m]) * 100
            hw_test = config["model_results"][m]["test_acc"] * 100
            acc_cols[i].metric(
                m,
                f"{acc:.1f}%",
                delta=f"Training test acc: {hw_test:.1f}%",
                delta_color="off",
            )

        # ---- Confusion matrices ----
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(len(MODEL_ORDER))
        for i, m in enumerate(MODEL_ORDER):
            with cm_cols[i]:
                fig_cm = plot_confusion_matrix(y_true_all, preds_all[m], m)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

        # ---- Precision / Recall / F1 ----
        st.subheader("Detailed Metrics")
        metrics_rows = []
        for m in MODEL_ORDER:
            p, r_val, f1, _ = precision_recall_fscore_support(
                y_true_all, preds_all[m], labels=[0, 1], average="binary", pos_label=1,
                zero_division=0,
            )
            acc = accuracy_score(y_true_all, preds_all[m])
            metrics_rows.append({
                "Model": m,
                "Accuracy": f"{acc:.3f}",
                "Precision": f"{p:.3f}",
                "Recall": f"{r_val:.3f}",
                "F1-Score": f"{f1:.3f}",
            })
        st.dataframe(
            pd.DataFrame(metrics_rows),
            use_container_width=True,
            hide_index=True,
        )

# -------------------------------------------------------------------
# TAB 6 – About / Method
# -------------------------------------------------------------------
with tabs[6]:
    st.header("About This App")

    st.markdown("""
    ### Project Overview

    This web application accompanies **Homework 3** on percussion-based
    delamination detection in composite plates. It provides a complete,
    interactive inference pipeline that uses the **trained models and
    preprocessing artifacts saved from the homework notebook**.

    ### Method

    1. **Audio Loading** – uploaded `.wav` files are read and converted to mono.
    2. **Segmentation** – a smoothed amplitude envelope is computed; peaks are
       detected using configurable thresholds; a window around each peak
       isolates individual hits.
    3. **Feature Extraction** – for each hit:
       - **PSD** (128 bins): Power Spectral Density via Welch's method,
         interpolated to a fixed number of frequency bins.
       - **MFCC** (13 mean + 13 std = 26 values): Mel-Frequency Cepstral
         Coefficients summarizing spectral shape.
       - Combined feature vector: **154 dimensions**.
    4. **Scaling** – the saved `StandardScaler` (fitted during training) is
       applied to normalize features.
    5. **Classification** – four models predict each hit independently:

    | Model | Test Accuracy |
    |-------|-------------------|
    | KNN | 69.0% |
    | Decision Tree | 53.4% |
    | Logistic Regression | 73.5% |
    | **SVM** | **78.8%** |

    ### Key Observations from Homework 3

    - **SVM** achieved the highest test accuracy (78.8%).
    - **Logistic Regression** had the smallest gap between validation and test
      accuracy, suggesting the best generalization behavior.
    - **Decision Tree** overfit significantly (100% train → 53.4% test).

    ### Configuration

    All segmentation and feature extraction parameters are loaded from
    `config.json`, ensuring perfect reproducibility with the homework pipeline.
    """)

    with st.expander("View config.json"):
        st.json(config)

    st.markdown("""
    ---
    ### Technical Stack
    - **Streamlit** – interactive web UI
    - **scikit-learn** – saved ML models and scaler
    - **librosa** – MFCC feature extraction
    - **scipy** – PSD computation and peak detection
    - **matplotlib** – publication-quality plots

    ### Repository Files Used
    | File | Purpose |
    |------|---------|
    | `config.json` | Segmentation & feature extraction settings |
    | `scaler.pkl` | Fitted StandardScaler |
    | `knn_model.pkl` | Trained KNN classifier |
    | `decision_tree_model.pkl` | Trained Decision Tree classifier |
    | `logistic_regression_model.pkl` | Trained Logistic Regression classifier |
    | `svm_model.pkl` | Trained SVM classifier |
    """)
