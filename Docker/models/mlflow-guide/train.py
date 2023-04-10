import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from mlflow.models.signature import infer_signature


# connects to the Mlflow tracking server that you started above
mlflow.set_tracking_uri("http://localhost:9005")
mlflow.set_experiment("mlflow_tracking_examples")

# loads the diabetes dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# run description (just metadata)
desc = "the simplest possible example"



# executes the run
with mlflow.start_run(run_name="logged_artifacts_with_signature") as run:
  params = {"n_estimators":100, "max_depth":6, "max_features":3}
  
  rf = RandomForestRegressor(**params)
  rf.fit(X_train, y_train)

  signature = infer_signature(X_train, rf.predict(X_test))
  input_example = X_train[0]

  mlflow.log_params(params)
  mlflow.sklearn.log_model(
      sk_model=rf,
      artifact_path="random_forest_regressor",
      input_example=input_example,
      signature=signature
  )