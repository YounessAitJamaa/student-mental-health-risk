"""Airflow DAG for the Machine Learning workflow."""

from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task
from src import ml_workflow


# Settings applied to all tasks
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}


@dag(
    dag_id="smhr_ml_pipeline",
    start_date=datetime(2026, 11, 15, tzinfo=timezone.utc),
    schedule="@weekly",
    catchup=False,
    default_args=default_args
)
def ml_pipeline():

    # Create the five Airflow tasks
    validate = task(ml_workflow.validate_data)
    prepare = task(ml_workflow.prepare_data)
    train = task(ml_workflow.train_model)
    evaluate = task(ml_workflow.evaluate_model)
    save = task(ml_workflow.save_artifacts)

    # Define task dependencies
    dataset = validate()
    prepared_data = prepare(dataset)
    trained_model = train(prepared_data)
    evaluated_model = evaluate(trained_model)
    save(evaluated_model)


# Register the DAG
ml_pipeline()