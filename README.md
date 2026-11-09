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