if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Create target
    if "ROAS" in data.columns:
        data["High_ROAS"] = (data["ROAS"] > 1).astype(int)
    else:
        st.error("ROAS column missing.")
        st.stop()

    # Drop unused columns EXACTLY like training
    drop_cols = ["ROAS", "date"]
    for col in drop_cols:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Separate features & target
    X = data.drop(columns=["High_ROAS"])
    y = data["High_ROAS"]

    # IMPORTANT: Ensure column order consistency
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    # Apply scaling only if needed
    if selected_model_name in ["Logistic Regression", "KNN"]:
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
        "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, auc, precision, recall, f1, mcc]
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
