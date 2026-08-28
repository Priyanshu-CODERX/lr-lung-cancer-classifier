import os
import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from logistic_regression import LogisticRegression
from data_utils import load_data, prepare_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def main():
    # === 1. Load and preprocess data ===
    data = load_data("../data/lung_cancer_preprocessed.csv")

    features = [
        "GENDER",
        "AGE",
        "SMOKING",
        "YELLOW_FINGERS",
        "ANXIETY",
        "PEER_PRESSURE",
        "CHRONIC_DISEASE",
        "FATIGUE",
        "ALLERGY",
        "WHEEZING",
        "ALCOHOL_CONSUMING",
        "COUGHING",
        "SHORTNESS_OF_BREATH",
        "SWALLOWING_DIFFICULTY",
        "CHEST_PAIN",
    ]

    target = "LUNG_CANCER"

    X, y = prepare_features(data, features, target)

    # === 2. Train/test split (stratified) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === 3. Feature scaling ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === 4. Train Logistic Regression ===
    model = LogisticRegression(lr=0.05, iterations=2000)
    model.fit(X_train, y_train)

    # === 5. Evaluate ===
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # === 6. Save trained model and scaler ===
    with open("../models/logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("../models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
