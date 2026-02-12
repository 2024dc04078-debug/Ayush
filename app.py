# ==========================================
# ML Assignment 2 - Streamlit App
# Global Ads Performance Classification
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


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(page_title="Global Ads Performance Classification App",
                   layout="wide")

st.title("📊 Global Ads Performance Classification App")
st.write("Predict whether an advertising campaign has High ROAS (>1).")


# ==========================================
# Available Models (Saved in Root Folder)
# ==========================================

available_models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}


# ==========================================
# Model Selection Dropdown
# ==========================================

selected_model_name = st.selectbox(
    "Select Model",
    list(available_models.keys())
)

model_filename = available_models[selected_model_name]

# ==========================================
# Load Selected Model
# ==========================================

if not os.path.exists(model_filename):
    st.error("Model file not found. Please ensure .pkl files are uploaded to GitHub.")
    st.stop()

with open(model_filename, "rb") as f:
    model = pickle.load(f)

# Load scaler
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None


# ==========================================
# Upload Test Dataset
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # ==========================================
    # Preprocessing (Same as Training)
    # =======
