# Percussion-Based Delamination Detection – Web App

A polished Streamlit web application that classifies percussion recordings from a composite plate as **Healthy** or **Unhealthy** (delaminated) using four pre-trained machine learning models from the Homework 3 pipeline.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## What This App Does

1. **Upload** one or more `.wav` percussion recordings
2. **Segment** multi-hit recordings into individual strikes automatically
3. **Extract features** (128 PSD bins + 26 MFCC features = 154 dimensions)
4. **Scale** features using the saved `StandardScaler`
5. **Classify** each hit with all four trained models:
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Logistic Regression
   - Support Vector Machine (SVM)
6. **Display** hit-level and file-level results with visualizations

## Filename Convention

If your `.wav` files follow this naming pattern, the app will automatically compute evaluation metrics:

| Suffix | Meaning |
|--------|---------|
| `*_g.wav` | **Good** (healthy / intact) |
| `*_b.wav` | **Bad** (unhealthy / delaminated) |

## App Tabs

| Tab | Description |
|-----|-------------|
| **Home** | Overview and pipeline explanation |
| **Upload & Process** | File upload and batch processing |
| **Segmentation Viewer** | Waveform, envelope, and detected peak visualizations |
| **Feature Viewer** | PSD and MFCC plots for individual hits |
| **Predictions** | Per-hit and per-file classification results with CSV export |
| **Evaluation** | Accuracy, confusion matrices, precision, recall, F1 (requires labeled files) |
| **About** | Methodology, model details, and configuration reference |

## Repository Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `pipeline.py` | Audio processing, feature extraction, and inference helpers |
| `config.json` | Segmentation and feature extraction parameters |
| `scaler.pkl` | Fitted StandardScaler from training |
| `knn_model.pkl` | Trained KNN classifier |
| `decision_tree_model.pkl` | Trained Decision Tree classifier |
| `logistic_regression_model.pkl` | Trained Logistic Regression classifier |
| `svm_model.pkl` | Trained SVM classifier |

## Model Performance (from Homework 3)

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| KNN | 98.3% | 97.4% | 69.0% |
| Decision Tree | 100.0% | 82.2% | 53.4% |
| Logistic Regression | 90.5% | 85.6% | 73.5% |
| **SVM** | **99.4%** | **95.1%** | **78.8%** |

- **SVM** achieved the best test accuracy
- **Logistic Regression** showed the smallest validation-to-test accuracy drop

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies
