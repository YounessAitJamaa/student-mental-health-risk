"""Small model-loading tests. Only temporary files are written."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import sklearn
from streamlit.testing.v1 import AppTest

from app.app import MODEL_PATH, load_prediction_model


class TestModelLoading(unittest.TestCase):
    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.joblib"
            with self.assertRaises(FileNotFoundError):
                load_prediction_model(missing_path, sklearn.__version__)

    def test_incompatible_version(self):
        with self.assertRaises(RuntimeError):
            load_prediction_model(MODEL_PATH, "0.0.0")

    def test_untrusted_path(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = Path(directory) / "untrusted.joblib"
            test_path.write_bytes(b"never load this file")
            with self.assertRaises(ValueError):
                load_prediction_model(test_path, sklearn.__version__)

    def test_corrupted_model(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = Path(directory) / "empty.joblib"
            test_path.write_bytes(b"")
            # Temporarily use a test file without touching the real saved model.
            with patch("app.app.MODEL_PATH", test_path):
                with self.assertRaises(EOFError):
                    load_prediction_model(test_path, sklearn.__version__)

    def test_invalid_model_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = Path(directory) / "invalid.joblib"
            joblib.dump({"description": "not a pipeline"}, test_path)
            with patch("app.app.MODEL_PATH", test_path):
                with self.assertRaises(TypeError):
                    load_prediction_model(test_path, sklearn.__version__)

    def test_streamlit_form_and_loading_error(self):
        app = AppTest.from_string("from app.app import main\nmain()")
        app.run(timeout=20)
        app.button[0].click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("Predicted dataset category:", app.info[-1].value)

        # One simple mock checks the error users see when loading fails.
        with patch("app.app.load_prediction_model", side_effect=OSError("private details")):
            with self.assertLogs("app.app", level="ERROR"):
                app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("prediction model is currently unavailable", app.error[0].value)
        self.assertNotIn("private details", app.error[0].value)
