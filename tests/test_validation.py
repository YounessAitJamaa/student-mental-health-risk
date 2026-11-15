import unittest
from unittest.mock import patch

import pandas as pd

from app.app import (
    MODEL_PATH,
    MODELS_DIR,
    get_model_categories,
    load_json,
    load_prediction_model,
    predict_student_profile,
    validate_numeric_inputs,
    validate_student_schema,
)


class TestStudentValidation(unittest.TestCase):

    def setUp(self):

        self.profile = {
            "Age": 21,
            "Gender": "Female",
            "Year_of_Study": 3,
            "Sleep_Hours": 7.0,
            "Study_Hours_Per_Week": 20.0,
            "GPA": 2.7,
            "Social_Media_Hours": 5.0,
            "Screen_Time_Hours": 8.0,
            "Physical_Activity": "High",
            "Financial_Stress": 5.0,
            "Family_Support": 7.0,
            "Attendance_Rate": 85.0,
            "Part_Time_Job": "No",
            "Diet_Quality": "Good",
            "Previous_Mental_Health_History": "No"
        }

        self.student_df = pd.DataFrame([self.profile])

        self.metadata = {
            "input_features": list(self.profile.keys())
        }

        self.categories = {
            "Gender": ["Female", "Male", "Non-binary"],
            "Physical_Activity": ["High", "Low", "Moderate"],
            "Part_Time_Job": ["No", "Yes"],
            "Diet_Quality": ["Average", "Good", "Poor"],
            "Previous_Mental_Health_History": ["No", "Yes"]
        }

    def test_valid_profile(self):

        valid_schema, _ = validate_student_schema(
            self.student_df,
            self.metadata,
            self.categories
        )

        valid_numeric, _ = validate_numeric_inputs(self.student_df)

        self.assertTrue(valid_schema)
        self.assertTrue(valid_numeric)

    def test_invalid_age(self):

        invalid_df = self.student_df.copy()
        invalid_df["Age"] = 150

        valid, _ = validate_numeric_inputs(invalid_df)

        self.assertFalse(valid)

    def test_missing_sleep(self):

        invalid_df = self.student_df.copy()
        invalid_df["Sleep_Hours"] = None

        valid, _ = validate_student_schema(
            invalid_df,
            self.metadata,
            self.categories
        )

        self.assertFalse(valid)

    def test_invalid_category(self):

        invalid_df = self.student_df.copy()
        invalid_df["Physical_Activity"] = "Very High"

        valid, _ = validate_student_schema(
            invalid_df,
            self.metadata,
            self.categories
        )

        self.assertFalse(valid)

    def test_missing_feature(self):

        invalid_df = self.student_df.drop(columns=["GPA"])

        valid, _ = validate_student_schema(
            invalid_df,
            self.metadata,
            self.categories
        )

        self.assertFalse(valid)

    def test_invalid_numeric_type(self):

        invalid_df = self.student_df.copy()
        invalid_df["GPA"] = "unknown"

        valid, _ = validate_numeric_inputs(invalid_df)

        self.assertFalse(valid)

    def test_invalid_numeric_edge_cases(self):
        for value in [float("nan"), float("inf"), True, 2 + 1j]:
            invalid_df = self.student_df.copy()
            invalid_df["GPA"] = value
            valid, _ = validate_numeric_inputs(invalid_df)
            self.assertFalse(valid)

        invalid_df = self.student_df.copy()
        invalid_df["Age"] = 21.5
        valid, _ = validate_numeric_inputs(invalid_df)
        self.assertFalse(valid)

    def test_invalid_dataframe_shapes(self):
        invalid_frames = [
            self.student_df.iloc[:0],
            pd.concat([self.student_df, self.student_df]),
            self.student_df.assign(Extra_Feature="fictional"),
            self.student_df[list(reversed(self.student_df.columns))],
        ]
        for invalid_df in invalid_frames:
            valid, _ = validate_student_schema(
                invalid_df, self.metadata, self.categories
            )
            self.assertFalse(valid)

    def test_saved_model_prediction(self):
        metadata = load_json(MODELS_DIR / "model_metadata.json")
        model = load_prediction_model(MODEL_PATH, metadata["sklearn_version"])
        categories = get_model_categories(model)

        result, message = predict_student_profile(
            model, self.student_df, metadata, categories
        )
        self.assertEqual(message, "")
        self.assertEqual(result, model.predict(self.student_df)[0])

    def test_invalid_inputs_are_rejected_before_prediction(self):
        metadata = load_json(MODELS_DIR / "model_metadata.json")
        model = load_prediction_model(MODEL_PATH, metadata["sklearn_version"])
        invalid_df = self.student_df.copy()
        invalid_df["Age"] = 150

        with patch.object(model, "predict") as predict:
            result, message = predict_student_profile(
                model, invalid_df, metadata, self.categories
            )
            predict.assert_not_called()

        self.assertIsNone(result)
        self.assertTrue(message)

    def test_inference_error_does_not_expose_inputs(self):
        metadata = load_json(MODELS_DIR / "model_metadata.json")
        model = load_prediction_model(MODEL_PATH, metadata["sklearn_version"])
        private_message = "fictional profile details"

        with patch.object(model, "predict", side_effect=RuntimeError(private_message)):
            with self.assertLogs("app.app", level="ERROR") as logs:
                result, message = predict_student_profile(
                    model, self.student_df, metadata, self.categories
                )

        self.assertIsNone(result)
        self.assertIn("Unable to generate a prediction", message)
        self.assertEqual(logs.records[0].getMessage(), "Model inference failed: RuntimeError")
        self.assertIsNone(logs.records[0].exc_info)
        self.assertNotIn(private_message, message)
