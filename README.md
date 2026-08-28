# Lung Cancer Predictor

A machine learning project that predicts the probability of lung cancer from patient demographic, lifestyle, and symptom features using a custom logistic regression model implemented from scratch in NumPy.

## Overview

This project implements an end-to-end ML pipeline:

1. **Data Preprocessing** — Encodes categorical features, scales numerical inputs, and handles class imbalance
2. **Model Training** — Custom logistic regression with gradient descent (no sklearn model dependency)
3. **Evaluation** — Achieves **91.94% test accuracy** with ROC-AUC analysis
4. **Web Interface** — Interactive Streamlit app for real-time predictions

## Project Structure

```
lung-cancer-predictor/
├── data/
│   ├── dataset.csv                        # Raw dataset (309 samples, 16 features)
│   └── lung_cancer_preprocessed.csv       # Cleaned and encoded dataset
├── models/
│   ├── logistic_model.pkl                 # Trained model weights
│   └── scaler.pkl                         # Fitted StandardScaler
├── notebooks/
│   └── data_preprocessing.ipynb           # EDA, preprocessing, and training notebook
├── src/
│   ├── app.py                             # Streamlit web application
│   ├── train.py                           # Training script
│   ├── logistic_regression.py             # Custom LogisticRegression class (NumPy)
│   └── data_utils.py                      # Data loading and feature preparation utilities
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

- **Custom Implementation** — Logistic regression built from scratch using NumPy (sigmoid, gradient descent, binary cross-entropy)
- **Interactive UI** — Streamlit app for predicting lung cancer risk from patient inputs
- **Comprehensive EDA** — Full exploratory analysis with correlation heatmaps, distribution plots, and feature importance
- **Reproducible** — Stratified train/test split with fixed random state

## Dataset

The dataset contains 309 samples with 15 features:

| Feature | Description |
|---------|-------------|
| `GENDER` | Male (1) / Female (0) |
| `AGE` | Patient age (21–87) |
| `SMOKING` | Smoking history |
| `YELLOW_FINGERS` | Yellow finger symptom |
| `ANXIETY` | Anxiety level |
| `PEER_PRESSURE` | Peer pressure influence |
| `CHRONIC_DISEASE` | Chronic disease presence |
| `FATIGUE` | Fatigue level |
| `ALLERGY` | Allergy presence |
| `WHEEZING` | Wheezing symptom |
| `ALCOHOL_CONSUMING` | Alcohol consumption |
| `COUGHING` | Coughing frequency |
| `SHORTNESS_OF_BREATH` | Breath shortness |
| `SWALLOWING_DIFFICULTY` | Swallowing issues |
| `CHEST_PAIN` | Chest pain presence |

**Target:** `LUNG_CANCER` (YES/NO)

## Results

| Metric | Class 0 (No) | Class 1 (Yes) |
|--------|-------------|---------------|
| Precision | 0.71 | 0.95 |
| Recall | 0.62 | 0.96 |
| F1-Score | 0.66 | 0.95 |

- **Training Accuracy:** 94.33%
- **Testing Accuracy:** 91.94%

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Priyanshu-CODERX/lr-lung-cancer-classifier.git
   cd lr-lung-cancer-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Train the model:**
```bash
cd src
python train.py
```

**Run the web app:**
```bash
cd src
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Model Details

- **Algorithm:** Logistic Regression (custom implementation)
- **Optimizer:** Gradient Descent
- **Learning Rate:** 0.05
- **Iterations:** 2,000
- **Loss Function:** Binary Cross-Entropy
- **Scaling:** StandardScaler (zero mean, unit variance)
- **Test Split:** 20% stratified

## License

This project is open source and available under the [MIT License](LICENSE).
