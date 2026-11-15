"""SMHR-27: two messages to check that Airflow works."""

from datetime import datetime, timezone

from airflow.sdk import DAG, task


@task
def welcome():
    print("Welcome to the Student Mental Health Risk project!")


@task
def complete():
    print("The Airflow example finished successfully.")


with DAG(
    dag_id="smhr_example",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,  # Run manually; there is no automatic schedule.
    catchup=False,  # Do not create runs for past dates.
):
    welcome_task = welcome()
    complete_task = complete()
    welcome_task >> complete_task  # Complete runs only after Welcome succeeds.
