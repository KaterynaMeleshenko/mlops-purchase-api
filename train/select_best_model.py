import os
import shutil
import mlflow


EXPERIMENT_NAME = "customer_purchase_catboost"
TARGET_DIR = "model"
TARGET_MODEL_NAME = "catboost_purchase_model.cbm"


def main():
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

    experiment_id = experiment.experiment_id

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1, 
    )

    if not runs:
        raise ValueError("No runs found in experiment")

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics["accuracy"]

    print(f"✅ Best run: {run_id}")
    print(f"✅ Best accuracy: {best_accuracy}")

    # Path to the model's artifacts in MLflow
    artifact_path = "model/catboost_purchase_model.cbm"

    os.makedirs(TARGET_DIR, exist_ok=True)

    # Download model from MLflow
    local_path = client.download_artifacts(run_id, artifact_path)

    # Popy it to the prod-folder model/
    shutil.copy(local_path, os.path.join(TARGET_DIR, TARGET_MODEL_NAME))

    print("✅ Best model copied to /model")


if __name__ == "__main__":
    main()