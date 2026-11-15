"""Run the five ML steps manually with Airflow."""

from airflow.sdk import dag, task
from src import ml_workflow


@dag(dag_id="smhr_ml_pipeline", schedule=None, catchup=False)
def ml_pipeline():
    # Turn the ordinary Python functions into Airflow tasks.
    validate = task(ml_workflow.validate_data)
    prepare = task(ml_workflow.prepare_data)
    train = task(ml_workflow.train_model)
    evaluate = task(ml_workflow.evaluate_model)
    save = task(ml_workflow.save_artifacts)

    # Each task receives a file path from the previous task.
    dataset = validate()
    prepared_data = prepare(dataset)
    trained_model = train(prepared_data)
    evaluated_model = evaluate(trained_model)
    save(evaluated_model)


ml_pipeline()
