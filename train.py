import os
import pickle
import click
import mlflow
from mlflow import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="/workspaces/MLOPs-Zoomcamp/output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):
    # Start MLflow run
    with mlflow.start_run():
        # Load data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Train model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # Calculate and log metrics
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        a = 10
        b = 0

        # Log parameters
        mlflow.log_params({
            "max_depth": a,
            "random_state": b
        })

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")

if __name__ == '__main__':
    run_train()
