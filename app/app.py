import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    st.set_page_config(
        page_title="Student Mental Health - Model Performance",
        layout="wide"
    )

    st.title("Student Mental Health Risk Prediction")

    st.warning(
        "Educational demonstration only. "
        "This application is not a clinical assessment or diagnosis tool."
    )

    try:
        metrics = load_json(REPORTS_DIR / "metrics.json")
        metadata = load_json(MODELS_DIR / "model_metadata.json")

        required_metrics = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc_ovr_macro",
            "classification_report",
            "labels"
        ]

        if not all(key in metrics for key in required_metrics):
            st.error("The evaluation metrics file is incomplete.")
            st.stop()

        class_results = pd.DataFrame({
            label: metrics["classification_report"][label]
            for label in metrics["labels"]
        }).T

    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Unable to load evaluation reports: {error}")
        st.stop()

    st.subheader("Selected Model")

    st.write("**Model:**", metadata.get("model_name", "Unknown"))
    st.write("**Best parameters:**", metadata.get("best_parameters", {}))

    st.subheader("Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision_macro']:.2%}")
    col3.metric("Recall", f"{metrics['recall_macro']:.2%}")
    col4.metric("Macro F1", f"{metrics['f1_macro']:.2%}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc_ovr_macro']:.3f}")

    st.subheader("Class-Specific Performance")

    st.dataframe(
        class_results[
            ["precision", "recall", "f1-score", "support"]
        ].round(3)
    )

    st.subheader("Model Comparison")

    comparison_path = REPORTS_DIR / "baseline_model_comparison.csv"

    try:
        comparison = pd.read_csv(comparison_path)

        st.dataframe(comparison)

        if "Model" in comparison.columns and "CV Mean (%)" in comparison.columns:
            st.bar_chart(
                comparison.set_index("Model")["CV Mean (%)"]
            )

    except (OSError, pd.errors.ParserError):
        st.warning("Model comparison results are unavailable.")

    st.subheader("Confusion Matrix")

    confusion_path = (
        REPORTS_DIR / "figures" / "test_confusion_matrix.png"
    )

    if confusion_path.exists():
        st.image(str(confusion_path))
    else:
        st.warning("Confusion matrix image is unavailable.")

    st.subheader("Key Findings")

    st.info(
        "The selected Logistic Regression model achieved "
        "75.5% accuracy and 0.754 Macro F1 on the independent test set. "
        "The Medium class was the most difficult to classify."
    )

    st.caption(
        "Dataset classification results do not establish clinical validity. "
        "This prototype must not be used to evaluate real students."
    )


if __name__ == "__main__":
    main()