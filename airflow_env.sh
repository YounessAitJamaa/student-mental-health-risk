# Run from the project root: source airflow_env.sh
source .venv-airflow/bin/activate

# Keep generated files separate from the code and ML environment.
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# Allow DAG tasks to import the project's src module.
export PYTHONPATH="$PWD"
mkdir -p "$AIRFLOW_HOME"
chmod 700 "$AIRFLOW_HOME"

# Make the development interface accessible only on this computer.
export AIRFLOW__API__HOST=127.0.0.1
export AIRFLOW__API__PORT=8080
