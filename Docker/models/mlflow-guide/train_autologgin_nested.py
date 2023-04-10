import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


# connect to mlflow
mlflow.set_tracking_uri("http://localhost:9005")
mlflow.set_experiment("mlflow_tracking_examples")


db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

mlflow.autolog(log_model_signatures=True, log_input_examples=True)

with mlflow.start_run(run_name="main_run_for_nested") as run:
  for estimators in range(20, 100, 20):
    with mlflow.start_run(run_name=f"nested_{estimators}_estimators", nested=True) as nested:
      rf = RandomForestRegressor(n_estimators=estimators, max_depth=6, max_features=3)
      rf.fit(X_train, y_train)