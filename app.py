import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")

st.markdown(
    "An interactive machine learning application "
    "for classifying seven varieties of dry beans."
)

st.divider()

# ------------------------------------------------
# 2. MODEL INFORMATION
# ------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

# ------------------------------------------------
# 3. SIDEBAR
# ------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.write("Dataset: UCI Dry Bean Dataset")
st.sidebar.write("Input Features: 16")
st.sidebar.write("Target Classes: 7")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_PATHS.keys())
)

st.sidebar.info(
    "Select a model and upload your test CSV "
    "to evaluate its classification performance."
)

# ------------------------------------------------
# 4. LOAD TEST DATA
# ------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    default_path = "test_data.csv"

    if os.path.exists(default_path):
        st.info("Using the default test dataset.")
        test_df = pd.read_csv(default_path)
    else:
        st.warning("Please upload a test CSV file.")
        st.stop()

st.success(
    f"Dataset loaded successfully: {test_df.shape[0]} records"
)

with st.expander("Preview uploaded dataset"):
    st.dataframe(test_df.head(10), use_container_width=True)

# ------------------------------------------------
# 5. CHECK DATA
# ------------------------------------------------

TARGET = "Class"

if TARGET not in test_df.columns:
    st.error(
        "The uploaded CSV must contain the actual target column: Class"
    )
    st.stop()

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ------------------------------------------------
# 6. EVALUATE MODEL
# ------------------------------------------------

st.header("2. Model Evaluation")

if st.button("Evaluate Selected Model", type="primary"):

    model_path = MODEL_PATHS[selected_model]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        model = joblib.load(model_path)

        missing_columns = [
            col for col in model.feature_names_in_
            if col not in X_test.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required features: {missing_columns}"
            )
            st.stop()

        X_test = X_test[list(model.feature_names_in_)]

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            y_test,
            y_prob,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        )

        precision = precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # ----------------------------------------
        # 7. DISPLAY METRICS
        # ----------------------------------------

        st.subheader(f"Results: {selected_model}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # ----------------------------------------
        # 8. CONFUSION MATRIX
        # ----------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(selected_model)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------
        # 9. CLASSIFICATION REPORT
        # ----------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

        # ----------------------------------------
        # 10. DOWNLOAD PREDICTIONS
        # ----------------------------------------

        output_df = test_df.copy()
        output_df["Predicted_Class"] = y_pred

        st.download_button(
            "Download Prediction Results",
            data=output_df.to_csv(index=False),
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

        st.success("Model evaluation completed successfully!")

    except Exception as e:
        st.error(f"Evaluation error: {e}")

# ------------------------------------------------
# 11. MODEL COMPARISON
# ------------------------------------------------

st.divider()

st.header("3. Model Comparison")

comparison_path = "results/model_comparison.csv"

if os.path.exists(comparison_path):

    comparison_df = pd.read_csv(comparison_path)

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    best_model = comparison_df.loc[
        comparison_df["F1"].idxmax(),
        "ML Model Name"
    ]

    st.success(
        f"Best model based on weighted F1 Score: {best_model}"
    )

else:
    st.info("Model comparison file is not available.")

st.divider()

st.caption(
    "Developed using Python, Scikit-learn and Streamlit | "
    "Dataset Source: UCI Machine Learning Repository"
)
