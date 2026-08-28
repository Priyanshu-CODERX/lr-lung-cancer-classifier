import streamlit as st
import pickle
import numpy as np
import pandas as pd

# === Load saved model and scaler ===
with open("../models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Streamlit UI ===
st.title("Lung Cancer Predictor")
st.write("Enter patient details to predict the probability of lung cancer:")

# Collect inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=50)

# List of binary features
binary_features = [
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

user_input = {}

# Arrange binary features in horizontal columns
n_cols = 4  # Number of columns per row
for i in range(0, len(binary_features), n_cols):
    cols = st.columns(n_cols)
    for j, feat in enumerate(binary_features[i : i + n_cols]):
        with cols[j]:
            user_input[feat] = st.radio(
                feat.replace("_", " ").title(), ["No", "Yes"], horizontal=True
            )

# Predict button
if st.button("Predict"):
    # Preprocess input
    input_array = []

    # Gender: Male=1, Female=0
    input_array.append(1 if gender == "Male" else 0)

    # Age
    input_array.append(age)

    # Binary features: Yes=1, No=0
    for feat in binary_features:
        input_array.append(1 if user_input[feat] == "Yes" else 0)

    # Convert to NumPy array and scale
    X_input = np.array(input_array).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Predict probability and label
    prob = model.predict_proba(X_scaled)  # returns a 1D array with 1 element
    prob_scalar = prob[0]  # get the scalar from array

    pred_label = "YES" if prob_scalar >= 0.5 else "NO"

    st.subheader("Prediction Results")
    st.write(f"Predicted Lung Cancer: **{pred_label}**")
    st.write(f"Probability of Lung Cancer: **{prob_scalar * 100:.2f}%**")
