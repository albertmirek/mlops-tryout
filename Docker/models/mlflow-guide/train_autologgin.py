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

with mlflow.start_run(run_name="autolog_with_named_run") as run:
  rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
  rf.fit(X_train, y_train)