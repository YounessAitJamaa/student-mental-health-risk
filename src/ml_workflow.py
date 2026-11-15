"""Machine Learning workflow for Apache Airflow."""

import logging
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/student_mental_health.csv"
METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"

logger = logging.getLogger(__name__)


# Load the model configuration
with open(METADATA_PATH, encoding="utf-8") as file:
    metadata = json.load(file)


# Task 1: Validate the dataset
def validate_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check required columns
    required_columns = metadata["input_features"] + [
        "Student_ID",
        metadata["target"]
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Check target values
    if not df[metadata["target"]].isin(metadata["target_classes"]).all():
        raise ValueError("Invalid target values.")

    # Check student identifiers
    if df["Student_ID"].isna().any():
        raise ValueError("Missing Student IDs.")

    if df["Student_ID"].duplicated().any():
        raise ValueError("Duplicate Student IDs.")

    print("Dataset validation completed!")

    return str(DATA_PATH)


# Task 2: Prepare training data
def prepare_data(dataset_path):

    df = pd.read_csv(dataset_path)

    # Separate features and target
    X = df[metadata["input_features"]]
    y = df[metadata["target"]]

    # Split the dataset
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        stratify=y
    )

    # Create a separate folder for this execution
    folder = PROJECT_ROOT / "data/processed/airflow"
    folder.mkdir(parents=True, exist_ok=True)

    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=folder))

    # Save training data for the next task
    joblib.dump((X_train, y_train), run_dir / "train_data.joblib")

    print("Training data prepared!")

    return str(run_dir)


# Task 3: Train the model
def train_model(run_directory):

    run_dir = Path(run_directory)

    # Load training data created by our workflow
    X_train, y_train = joblib.load(run_dir / "train_data.joblib")

    # Identify numerical and categorical features
    numerical_columns = X_train.select_dtypes(
        include="number"
    ).columns

    categorical_columns = X_train.select_dtypes(
        include=["str", "object", "category"]
    ).columns

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    # Create the complete Machine Learning pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            C=metadata["best_parameters"]["classifier__C"],
            max_iter=1000,
            random_state=metadata["random_state"]
        ))
    ])

    # Train on training data only
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, run_dir / "model.joblib")

    logger.info("Model training completed!")

    return str(run_dir)


# Task 4: Evaluate the model
def evaluate_model(run_directory):

    run_dir = Path(run_directory)

    # Load training data and model
    X_train, y_train = joblib.load(run_dir / "train_data.joblib")
    model = joblib.load(run_dir / "model.joblib")

    # Configure cross-validation
    cv = StratifiedKFold(
        n_splits=metadata["cv_folds"],
        shuffle=True,
        random_state=metadata["random_state"]
    )

    # Evaluate using training data only
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=metadata["scoring"]
    )

    # Store evaluation results
    metrics = {
        "model": "Logistic Regression",
        "scoring": metadata["scoring"],
        "cv_folds": metadata["cv_folds"],
        "cv_macro_f1": float(scores.mean()),
        "cv_std": float(scores.std()),
        "test_set_used": False,
        "educational_only": True
    }

    # Save the metrics
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model evaluation completed!")
    logger.info("CV Macro F1: %.4f", scores.mean())

    return str(run_dir)


# Task 5: Save the final artifacts
def save_artifacts(run_directory):

    run_dir = Path(run_directory)

    # Create separate folders for Airflow results
    model_dir = PROJECT_ROOT / "models/airflow" / run_dir.name
    report_dir = PROJECT_ROOT / "reports/airflow" / run_dir.name

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Copy the model
    shutil.copy2(
        run_dir / "model.joblib",
        model_dir / "logistic_regression_pipeline.joblib"
    )

    # Copy the evaluation results
    shutil.copy2(
        run_dir / "metrics.json",
        report_dir / "metrics.json"
    )

    print("Model and evaluation results saved!")

    return str(report_dir)