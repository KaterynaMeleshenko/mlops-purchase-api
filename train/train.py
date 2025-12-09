import os

import mlflow
import mlflow.catboost

from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Load data

def load_data():
    df = pd.read_csv("data/customer_purchase.csv")
    X = df.drop(columns=["PurchaseStatus"])
    y = df["PurchaseStatus"]
    return train_test_split(X, y, test_size=0.2, random_state=0)


# 2. Hyperparameters

param_dist = {
    "iterations": randint(50, 500),
    "depth": randint(4, 10),
    "learning_rate": uniform(0.01, 0.3),
    "l2_leaf_reg": uniform(1, 10),
    "border_count": randint(32, 255),
    "bagging_temperature": uniform(0, 1),
    "random_strength": uniform(1, 20),
    "one_hot_max_size": randint(2, 10),
    "rsm": uniform(0.5, 1.0),
}


def train():
    X_train, X_test, y_train, y_test = load_data()

    cat_model = CatBoostClassifier(random_state=0, silent=True)
    random_search = RandomizedSearchCV(
        estimator=cat_model,
        param_distributions=param_dist,
        n_iter=10,   
        cv=3,
        verbose=2,
        random_state=0,
        n_jobs=-1,
    )

    # 3. Start experiment in MLflow
    mlflow.set_experiment("customer_purchase_catboost")

    with mlflow.start_run():
        random_search.fit(X_train, y_train)

        best_cat_model = random_search.best_estimator_

        y_pred = best_cat_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Best params:", random_search.best_params_)
        print("Accuracy:", accuracy)

        # 4. Log hyperparameters
        mlflow.log_params(random_search.best_params_)

        # 5. Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # 6. Save model as a file and as a MLflow-model

        os.makedirs("model", exist_ok=True)
        model_path = "model/catboost_purchase_model.cbm"
        best_cat_model.save_model(model_path, format="cbm")

        # as an artifact
        mlflow.log_artifact(model_path, artifact_path="model")

        # as MLflow-model
        mlflow.catboost.log_model(
            cb_model=best_cat_model,
            artifact_path="catboost-model"
        )


if __name__ == "__main__":
    train()