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

## Apache Airflow Development Setup (SMHR-27)

Airflow 3.4.0 supports Python 3.14. This setup uses a separate `.venv-airflow`
environment so it does not change the ML packages in `.venv`.
See the official [prerequisites](https://airflow.apache.org/docs/apache-airflow/stable/installation/prerequisites.html)
and [installation guide](https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html).

Run these commands from the project root. Create the environment once:

```bash
.venv/bin/python -m venv .venv-airflow
source airflow_env.sh
python -m pip install -r requirements-airflow.txt
airflow version
python -m pip check
```

`requirements-airflow.txt` pins Airflow 3.4.0 and uses its official Python 3.14
constraints file, which selects compatible dependency versions. Do not install
this file into the ML `.venv`.

For every new terminal used for Airflow, run `source airflow_env.sh` from the
project root. This activates the Airflow environment and sets:

- `AIRFLOW_HOME` to `.airflow/` for the local database, logs, and credentials.
- `AIRFLOW__CORE__DAGS_FOLDER` to the project's `dags/` directory.
- `AIRFLOW__CORE__LOAD_EXAMPLES=False` to hide the bundled example DAGs.
- The API server host and port to `127.0.0.1:8080` for local access.

The script also creates `.airflow/` with access restricted to your Linux user.
Both `.airflow/` and `.venv-airflow/` are ignored by Git. Never share or commit
generated passwords or logs.

Start Airflow in the first terminal and leave it running:

```bash
source airflow_env.sh
airflow standalone
```

Standalone initializes a SQLite database and starts the API server (including
the web interface), scheduler, DAG processor, and triggerer. This is a local
development setup, not a production deployment. Open <http://127.0.0.1:8080>
and log in with `admin`. On first startup, Airflow prints a generated password
in your own terminal. On later startups it does not print that password again.
You can privately open `.airflow/simple_auth_manager_passwords.json.generated`
to find it. This ignored file was already created during setup verification;
never paste its contents into Git or shared presentation slides.

In a second terminal, check DAG discovery and run the example:

```bash
source airflow_env.sh
airflow dags list
airflow dags list-import-errors
airflow dags unpause smhr_example
airflow dags trigger smhr_example
airflow dags list-runs smhr_example
```

To check individual task statuses, copy a run ID from the last command:

```bash
airflow tasks states-for-dag-run smhr_example YOUR_RUN_ID
```

Allow a little time for the DAG processor to discover a new file and for the
scheduler to execute a triggered run. In the web interface, open `smhr_example`,
select its latest run, and check that both tasks are green (`success`). Open
each task's logs to see the welcome and completion messages. You can also
trigger another run using the interface's **Trigger DAG** button.

The example in `dags/smhr_example.py` contains only `welcome` and `complete`.
`@task` tells Airflow to run each Python function as a task. The `with DAG(...)`
block groups them into `smhr_example`. `schedule=None` makes it manual, and
`catchup=False` prevents historical runs. Calling the decorated functions
creates the tasks; `welcome_task >> complete_task` makes the second wait for
the first to succeed. It does not load data or train a model. The ML workflow
is reserved for SMHR-28.

To stop all standalone services, press **Ctrl+C** in the first terminal and
wait for shutdown. Run `deactivate` to leave the Airflow environment. To return
to the ML environment, run `source .venv/bin/activate`.

Setup was verified locally on Python 3.14.8 with Airflow 3.4.0: dependency
checks passed, no DAG import errors were found, and run `smhr27_verification`
completed both tasks successfully with the expected log messages. The web
page and authenticated DAG API were reachable, and all four service health
checks passed. All 17 existing ML tests also passed. Services were stopped
after verification; use the start command above for your presentation.

Airflow's CLI may display upstream Python deprecation warnings and a warning
about the optional Graphviz package. Graphviz is not needed to run this DAG or
use the web interface; no extra graph-rendering dependency was installed.
