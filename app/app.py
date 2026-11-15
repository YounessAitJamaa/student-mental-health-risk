import json
from pathlib import Path

import pandas as pd
import streamlit as st

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "final_logistic_regression_pipeline.joblib"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_resource
def load_prediction_model():
    model = joblib.load(MODEL_PATH)
    return model


def show_prediction_form(model, metadata):

    preprocessor = model.named_steps["preprocessing"]

    categorical_columns = preprocessor.transformers_[1][2]

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    categories = dict(zip(categorical_columns, encoder.categories_))

    st.info("Use fictional student information only.")

    with st.form("student_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=17,
                max_value=29,
                value=21
            )

            year = st.number_input(
                "Year of Study",
                min_value=1,
                max_value=5,
                value=3
            )

            sleep = st.number_input(
                "Sleep Hours",
                min_value=0.0,
                max_value=12.0,
                value=7.0
            )

            study = st.number_input(
                "Study Hours Per Week",
                min_value=0.0,
                max_value=80.0,
                value=20.0
            )

            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=2.7
            )

            social_media = st.number_input(
                "Social Media Hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0
            )

            screen_time = st.number_input(
                "Screen Time Hours",
                min_value=0.0,
                max_value=24.0,
                value=8.0
            )

            attendance = st.number_input(
                "Attendance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                categories["Gender"]
            )

            physical_activity = st.selectbox(
                "Physical Activity",
                categories["Physical_Activity"]
            )

            financial_stress = st.number_input(
                "Financial Stress",
                min_value=1.0,
                max_value=10.0,
                value=5.0
            )

            family_support = st.number_input(
                "Family Support",
                min_value=1.0,
                max_value=10.0,
                value=7.0
            )

            part_time_job = st.selectbox(
                "Part-Time Job",
                categories["Part_Time_Job"]
            )

            diet_quality = st.selectbox(
                "Diet Quality",
                categories["Diet_Quality"]
            )

            previous_history = st.selectbox(
                "Previous Mental Health History",
                categories["Previous_Mental_Health_History"]
            )

        submitted = st.form_submit_button("Test Fictional Profile")

    if submitted:

        student_data = {
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study,
            "GPA": gpa,
            "Social_Media_Hours": social_media,
            "Screen_Time_Hours": screen_time,
            "Physical_Activity": physical_activity,
            "Financial_Stress": financial_stress,
            "Family_Support": family_support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": previous_history
        }

        expected_features = metadata["input_features"]

        if set(student_data.keys()) != set(expected_features):
            st.error("The submitted features do not match the model requirements.")
            return None

        student_df = pd.DataFrame(
            [student_data],
            columns=expected_features
        )

        # Generate prediction
        try:
            prediction = model.predict(student_df)

            predicted_class = prediction[0]

        except (ValueError, TypeError):
            st.error("Unable to generate a prediction. Please check the input values.")
            return None


        # Validate predicted class
        if predicted_class not in metadata["target_classes"]:
            st.error("The model returned an unexpected category.")
            return None


        # Display prediction
        st.subheader("Educational Classification Result")

        st.info(f"Predicted dataset category: {predicted_class}")

        st.warning(
            "Educational demonstration using fictional data only. "
            "This prediction is not a medical diagnosis, clinical assessment, "
            "or a reliable measure of an individual's mental health."
        )

        return student_df

        st.success("Fictional profile submitted successfully!")
   
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

    st.subheader("Fictional Student Prediction")

    try:
        prediction_model = load_prediction_model()

        st.success("Prediction model loaded successfully!")

        st.write("Pipeline steps:", list(prediction_model.named_steps))

        student_df = show_prediction_form(prediction_model, metadata)

    except (OSError, ValueError) as error:
        st.error(f"Unable to load prediction model: {error}")


if __name__ == "__main__":
    main()