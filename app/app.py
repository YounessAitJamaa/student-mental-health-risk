import json
import logging
import math
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "final_logistic_regression_pipeline.joblib"

NUMERIC_RANGES = {
    "Age": (17, 29),
    "Year_of_Study": (1, 5),
    "Sleep_Hours": (0, 12),
    "Study_Hours_Per_Week": (0, 80),
    "GPA": (0, 4),
    "Social_Media_Hours": (0, 24),
    "Screen_Time_Hours": (0, 24),
    "Financial_Stress": (1, 10),
    "Family_Support": (1, 10),
    "Attendance_Rate": (0, 100)
}

CATEGORICAL_FEATURES = [
    "Gender", "Physical_Activity", "Part_Time_Job", "Diet_Quality",
    "Previous_Mental_Health_History"
]


def load_json(file_path):
    """Read a JSON report or metadata file into a Python dictionary."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_prediction_model(model_path, expected_version):
    """Load only the project's trusted, locally installed model artifact."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError("Model artifact is missing.")

    # Joblib can execute code. Never accept an uploaded or user-selected file.
    if model_path.resolve() != MODEL_PATH.resolve() or model_path.is_symlink():
        raise ValueError("Only the trusted project model can be loaded.")

    if sklearn.__version__ != expected_version:
        raise RuntimeError("Incompatible Scikit-learn version.")

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Expected a Scikit-learn pipeline.")

    for step in ["preprocessing", "classifier"]:
        if step not in model.named_steps:
            raise ValueError("The model pipeline is incomplete.")
    if not hasattr(model, "predict"):
        raise TypeError("The model cannot make predictions.")

    # Make sure the saved categories needed by the form are available.
    get_model_categories(model)

    return model


def get_model_categories(model):
    """Read the dropdown choices from the saved categorical encoder."""
    preprocessor = model.named_steps["preprocessing"]
    categories = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["encoder"]
            for index, feature in enumerate(columns):
                categories[feature] = encoder.categories_[index]

    for feature in CATEGORICAL_FEATURES:
        if feature not in categories or len(categories[feature]) == 0:
            raise ValueError("Model categories are missing.")

    return categories


def show_prediction_form(model, metadata):
    """Show the 15 inputs and display a result after successful validation."""
    categories = get_model_categories(model)

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

        if list(student_data) != expected_features:
            st.error("The submitted features do not match the model requirements.")
            return None

        student_df = pd.DataFrame(
            [student_data],
            columns=expected_features
        )

        predicted_class, error_message = predict_student_profile(
            model, student_df, metadata, categories
        )
        if error_message:
            st.error(error_message)
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


def predict_student_profile(model, student_df, metadata, categories):
    """Validate one fictional profile, returning a result or a simple error."""
    try:
        valid, message = validate_student_schema(student_df, metadata, categories)
        if not valid:
            return None, message

        valid, message = validate_numeric_inputs(student_df)
        if not valid:
            return None, message

        prediction = model.predict(student_df)
        if len(prediction) != 1:
            raise ValueError("Expected exactly one prediction.")
        predicted_class = prediction[0]
        if predicted_class not in metadata["target_classes"]:
            raise ValueError("Unexpected prediction category.")
        return predicted_class, ""

    except Exception as error:
        # Exception messages and tracebacks may contain submitted feature values.
        logger.error("Model inference failed: %s", type(error).__name__)
        return None, "Unable to generate a prediction. Please try again later."


def validate_numeric_inputs(student_df):
    """Check numeric types, finite values, ranges, and whole-number fields."""
    if not isinstance(student_df, pd.DataFrame):
        return False, "Input must be a Pandas DataFrame."
    if len(student_df) != 1:
        return False, "Please submit exactly one fictional profile."
    if student_df.columns.has_duplicates:
        return False, "Input features must not be duplicated."

    for feature, (minimum, maximum) in NUMERIC_RANGES.items():

        if feature not in student_df.columns:
            return False, f"Missing required feature: {feature}"

        value = student_df[feature].iloc[0]

        # Real accepts Python and NumPy numbers; bool must be rejected separately.
        if not isinstance(value, Real) or isinstance(value, bool):
            return False, f"{feature} must be a number."

        if not math.isfinite(float(value)):
            return False, f"{feature} must be a finite number."

        if value < minimum or value > maximum:
            return False, f"{feature} must be between {minimum} and {maximum}."

        if feature in ("Age", "Year_of_Study") and not float(value).is_integer():
            return False, f"{feature} must be a whole number."

    return True, ""


def validate_student_schema(student_df, metadata, categories):
    """Check one complete row, the feature order, and the allowed categories."""

    expected_features = metadata["input_features"]

    if not isinstance(student_df, pd.DataFrame):
        return False, "Input must be a Pandas DataFrame."

    if len(student_df) != 1:
        return False, "Please submit exactly one fictional profile."

    if list(student_df.columns) != expected_features:
        return False, "Input features do not match the model schema."

    if student_df.isna().any().any():
        return False, "All input fields are required."

    for feature, allowed_values in categories.items():

        value = student_df[feature].iloc[0]

        if not isinstance(value, str):
            return False, f"{feature} must be a category."

        if value not in allowed_values:
            return False, f"Invalid category for {feature}."

    return True, ""


def main():
    """Display the saved evaluation results and the fictional-profile form."""
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
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a dictionary.")

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
        logger.error("Report or metadata loading failed: %s", type(error).__name__)
        st.error("The evaluation reports or model information are currently unavailable.")
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
        prediction_model = load_prediction_model(
            MODEL_PATH,
            metadata["sklearn_version"]
        )
        if list(prediction_model.feature_names_in_) != metadata["input_features"]:
            raise ValueError("Model features do not match the metadata.")

    except Exception as error:

        logger.error(
            "Model loading failed: %s",
            type(error).__name__
        )

        st.error(
            "The prediction model is currently unavailable. "
            "Please verify the model installation."
        )

    else:
        show_prediction_form(prediction_model, metadata)


if __name__ == "__main__":
    main()
