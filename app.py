import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Global Ads Performance Classification App",
                   layout="wide")

st.title("📊 Global Ads Performance Classification App")
st.write("Predict whether an advertising campaign has High ROAS (>1).")

# -------------------------------
# Model Selection
# -------------------------------
available_models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

selected_model_name = st.selectbox(
    "Select Model",
    list(available_models.keys())
)

model_filename = available_models[selected_model_name]

if not os.path.exists(model_filename):
    st.error("Model file not found. Upload .pkl files to GitHub.")
    st.stop()

with open(model_filename, "rb") as f:
    model = pickle.load(f)

# Load scaler
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Create Target
    if "ROAS" not in data.columns:
        st.error("ROAS column missing.")
        st.stop()

    data["High_ROAS"] = (data["ROAS"] > 1).astype(int)

    # Drop columns same as training
    for col in ["ROAS", "date"]:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # Encode categoricals
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data.drop(columns=["High_ROAS"])
    y = data["High_ROAS"]

    # Ensure same column order
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    # Scaling if needed
    if selected_model_name in ["Logistic Regression", "KNN"]:
        if scaler is None:
            st.error("Scaler file missing.")
            st.stop()
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    # Predictions
    y_pred = model.predict(X_processed)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_processed)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = None

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("📈 Evaluation Metrics")

    st.table(pd.DataFrame({
        "Metric": ["Accuracy", "AUC Score", "Precision",
                   "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, auc, precision,
                  recall, f1, mcc]
    }))

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.info("Please upload a CSV file to evaluate the selected model.")
