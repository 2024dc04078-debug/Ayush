# ==========================================
# ML Assignment 2 - Streamlit Application
# ==========================================

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


st.set_page_config(page_title="Ad Performance Classifier", layout="wide")

st.title("📊 Global Ads Performance Classification App")
st.write("Predict whether an advertising campaign has High ROAS (>1).")


# ==========================================
# 1. Load Saved Models
# ==========================================

MODEL_PATH = "model"

available_models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}


# ==========================================
# 2. Model Selection
# ==========================================

selected_model_name = st.selectbox(
    "Select Model",
    list(available_models.keys())
)

model_file = available_models[selected_model_name]

if os.path.exists(os.path.join(MODEL_PATH, model_file)):
    with open(os.path.join(MODEL_PATH, model_file), "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please ensure models are trained.")
    st.stop()


# ==========================================
# 3. Upload Test Dataset
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    # ===============================
    # Preprocessing (same as training)
    # ===============================

    if "ROAS" in data.columns:
        data["High_ROAS"] = (data["ROAS"] > 1).astype(int)
        data.drop(columns=["ROAS"], inplace=True)

    if "date" in data.columns:
        data.drop(columns=["date"], inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    if "High_ROAS" not in data.columns:
        st.error("Target column could not be created.")
        st.stop()

    X = data.drop(columns=["High_ROAS"])
    y = data["High_ROAS"]

    # ===============================
    # Make Predictions
    # ===============================

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = "Not Available"

    # ===============================
    # Evaluation Metrics
    # ===============================

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("📈 Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, auc, precision, recall, f1, mcc]
    })

    st.table(metrics_df)

    # ===============================
    # Confusion Matrix
    # ===============================

    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    st.pyplot(fig)

    # ===============================
    # Classification Report
    # ===============================

    st.subheader("📄 Classification Report")
    report = classification_report(y, y_pred)
    st.text(report)

else:
    st.info("Please upload a CSV test file to proceed.")
