import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 10px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Bank Marketing Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare multiple machine learning classification models '
    'for predicting term-deposit subscription.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_DIR = "Models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(model_path):

            loaded_models[model_name] = joblib.load(
                model_path
            )

    return loaded_models


models = load_models()

# ============================================================
# CHECK MODEL FILES
# ============================================================

if not models:

    st.error(
        "No trained models were found. "
        "Please make sure the 'models' folder contains "
        "the saved .pkl files."
    )

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Application Settings")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Target Variable**

    `deposit`

    **0 → No Deposit**

    **1 → Deposit**
    """
)

# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload `test_data.csv` to evaluate the selected model."
    )

    st.markdown(
        """
        ### Required CSV format

        The uploaded CSV should contain the 16 input features
        and the target column `deposit`.

        Example target values:

        - `yes`
        - `no`
        """
    )

    st.stop()

# ============================================================
# READ CSV
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the CSV file: {e}")

    st.stop()

# ============================================================
# DISPLAY DATA
# ============================================================

st.subheader("📊 Uploaded Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        data.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        data.shape[1]
    )

with col3:
    st.metric(
        "Selected Model",
        selected_model
    )

with st.expander("View Uploaded Data"):

    st.dataframe(
        data,
        use_container_width=True
    )

# ============================================================
# VALIDATE TARGET
# ============================================================

TARGET = "deposit"

if TARGET not in data.columns:

    st.error(
        "The uploaded CSV must contain the "
        "`deposit` target column."
    )

    st.stop()

# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = data.drop(
    columns=[TARGET]
)

y_test = data[TARGET].map(
    {
        "no": 0,
        "yes": 1
    }
)

# Check for invalid target values

if y_test.isnull().any():

    st.error(
        "The `deposit` column must contain only "
        "`yes` or `no` values."
    )

    st.stop()

# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model = models[selected_model]

# ============================================================
# MAKE PREDICTIONS
# ============================================================

try:

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()

# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

# ============================================================
# DISPLAY METRICS
# ============================================================

st.divider()

st.header("📈 Model Evaluation")

st.subheader(
    f"Performance of {selected_model}"
)

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with metric2:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with metric3:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


metric4, metric5, metric6 = st.columns(3)

with metric4:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with metric5:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with metric6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.header("🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Deposit", "Actual: Deposit"],
    columns=["Predicted: No Deposit", "Predicted: Deposit"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Deposit",
        "Deposit"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.divider()

st.header("🎯 Prediction Summary")

prediction_col1, prediction_col2 = st.columns(2)

with prediction_col1:

    no_deposit_count = np.sum(
        y_pred == 0
    )

    st.metric(
        "Predicted No Deposit",
        int(no_deposit_count)
    )

with prediction_col2:

    deposit_count = np.sum(
        y_pred == 1
    )

    st.metric(
        "Predicted Deposit",
        int(deposit_count)
    )

# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header("🏆 Model Comparison")

RESULTS_FILE = "model_results.csv"

if os.path.exists(RESULTS_FILE):

    comparison_df = pd.read_csv(
        RESULTS_FILE
    )

    comparison_display = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    for column in metric_columns:

        if column in comparison_display.columns:

            comparison_display[column] = (
                comparison_display[column]
                .round(4)
            )

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )

    # Find overall best model based on average of metrics

    available_metrics = [
        col
        for col in metric_columns
        if col in comparison_df.columns
    ]

    comparison_df["Average Score"] = (
        comparison_df[available_metrics]
        .mean(axis=1)
    )

    winner = comparison_df.loc[
        comparison_df["Average Score"].idxmax()
    ]

    st.success(
        f"🏆 Overall Best Model: "
        f"**{winner['ML Model Name']}** "
        f"(Average Score: "
        f"{winner['Average Score']:.4f})"
    )

else:

    st.warning(
        "model_results.csv was not found. "
        "Model comparison is unavailable."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Assignment 2 | "
    "Bank Marketing Classification"
)
