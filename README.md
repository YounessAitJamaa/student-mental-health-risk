# Student Mental Health Risk Prediction

A Machine Learning project focused on predicting student mental health risk categories using supervised classification algorithms.

## Project Objectives

- Explore and analyze the dataset.
- Clean and preprocess the data.
- Train and compare classification models.
- Evaluate model performance.
- Automate the ML workflow using Apache Airflow.
- Build an interactive application using Streamlit.

## Project Scope

This project focuses on building an educational Machine Learning classification system using a student mental health dataset.

The project includes:
- Data exploration and visualization.
- Data cleaning and preprocessing.
- Training and comparing three classification models.
- Model evaluation and optimization.
- Workflow automation using Apache Airflow.
- An interactive demonstration using Streamlit.

The project does not provide medical diagnoses or real-world mental health assessments. Predictions will be demonstrated using fictional student profiles only.

## Machine Learning Workflow

The project follows these steps:

1. **Data Exploration:** Load the dataset, understand its features, and visualize patterns.
2. **Data Preprocessing:** Handle missing values, categorical features, and numerical scaling while preventing data leakage.
3. **Model Training:** Train Logistic Regression, Decision Tree, and Random Forest classifiers.
4. **Model Evaluation:** Compare models using classification metrics and select the best-performing approach.
5. **Model Optimization:** Tune hyperparameters using cross-validation.
6. **Model Saving:** Save the trained preprocessing and classification pipeline using Joblib.
7. **Workflow Automation:** Organize and execute the ML pipeline using Apache Airflow.
8. **Application:** Build a Streamlit dashboard to display results and demonstrate predictions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Apache Airflow
- Streamlit

## Project Structure

The project is organized into the following directories:

- `data/` — Raw and processed datasets.
- `notebooks/` — Jupyter notebooks for data exploration and experiments.
- `src/` — Python scripts for preprocessing, training, and evaluation.
- `dags/` — Apache Airflow DAG definitions.
- `app/` — Streamlit application.
- `models/` — Saved Machine Learning models.
- `reports/` — Evaluation results and visualizations.
- `tests/` — Python tests.

## Dataset

Source: [Kaggle - Mental Health Risk Prediction Dataset](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)

## Project Management

Project planning and task tracking are managed using Jira.

[Jira Project — Student Mental Health Risk](https://aitjamaayouness10.atlassian.net/jira/software/projects/SMHR/summary)

## Results and Screenshots

Model evaluation results, application screenshots, and Airflow execution evidence will be added as the project develops.

## Installation

Clone the repository:

```bash
git clone https://github.com/YounessAitJamaa/student-mental-health-risk.git
cd student-mental-health-risk
```

Create and activate a virtual environment (Linux):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Start JupyterLab:

```bash
jupyter lab
```

Run the environment test script:

```bash
python src/check_environment.py
```

## Note

This project is for educational purposes only. Its predictions are not medical diagnoses or suitable for real-world student mental health assessments.

## Model Artifacts

The final selected model is Logistic Regression.

The complete fitted preprocessing and classification pipeline
is saved in:

- `models/final_logistic_regression_pipeline.joblib`

Additional artifacts:

- `models/model_metadata.json` — Model configuration, input features,
  target classes, and dependency versions.
- `reports/metrics.json` — Independent test evaluation results.
- `reports/baseline_model_comparison.csv` — Initial model comparison.
- `reports/model_comparison.csv` — Hyperparameter optimization results.
- `reports/figures/test_confusion_matrix.png` — Confusion matrix.

The serialized pipeline includes missing-value imputation,
feature scaling, categorical encoding, and classification.

Model inference requires the 15 input features documented
in `model_metadata.json`.

The model should be loaded only from trusted files using
compatible Python and Scikit-learn dependencies.

Predictions are intended for educational demonstration
using fictional profiles only.

## Streamlit Application

The project includes an interactive dashboard for exploring
Machine Learning classification performance.

### Features

- Final model performance metrics.
- Class-specific classification results.
- Model comparison visualizations.
- Confusion matrix.
- Educational-use disclaimer.

The application loads previously generated evaluation artifacts
and does not retrain the Machine Learning model.

### Run the Application

Activate the Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt

```
### Fictional Student Prediction

The Streamlit application includes an interactive prediction form
for fictional student profiles.

The form collects the 15 features required by the trained model.

The application loads the saved Machine Learning pipeline using Joblib
and uses Streamlit caching to avoid unnecessary model reloading.

Predictions are generated without retraining the model.

The application does not intentionally save or log submitted profiles.

This feature is strictly educational and must not be used for
clinical assessment or real student evaluation.

### Input Validation and Error Handling (SMHR-33)

Before prediction, the application checks that exactly one fictional profile
contains all 15 features in the expected order, with no missing or extra fields.
Numeric values must be finite and within the form's allowed ranges; age and
year of study must be whole numbers. Categories must match the fitted encoder.
Invalid profiles are rejected before the model is called.

The model is loaded with `st.cache_resource` only from the trusted project
artifact. Never replace it with an untrusted file: Joblib deserialization can
execute code. The loader checks file availability, the exact Scikit-learn
version recorded in `models/model_metadata.json`, the required pipeline steps,
and the dropdown categories. The app also checks the model's feature order
against the metadata. Use the project's pinned dependencies; the app never
retrains the model.

Missing, corrupted, incompatible, or incomplete models show a simple
unavailable message. Unexpected prediction failures also show a simple error.
Technical logging records only the failed operation and exception type,
without exception messages, tracebacks, submitted profiles, or feature values.
The application does not save submitted profiles or configure a log file.
After repairing or replacing a trusted artifact, restart Streamlit to clear
the cached model.

Run the application from the project root:

```bash
python -m streamlit run app/app.py
```

Run the tests after activating the virtual environment and installing dependencies:

```bash
python -m unittest discover -s tests -v
```

The small test suite keeps the six original validation tests. Additional
tests cover invalid inputs, model-loading failures, safe inference errors,
and the Streamlit form. They use fictional profiles, temporary files, and a
few simple mocks. The real saved pipeline is read without changing it.
This demonstration has no established clinical validity: predictions are
dataset categories, not diagnoses or reliable assessments of real students.
