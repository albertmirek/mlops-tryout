import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlflow.models.signature import infer_signature


import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:9005")
mlflow.set_experiment("on-boarding-scikit-boston-housing")

if __name__ == "__main__":
  #Load data
  #boston = datasets.load_boston() 
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
  data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  target = raw_df.values[1::2, 2]

  df = pd.DataFrame(data)
  df["MEDV"] = target

  #Split Model
  X = df.drop(["MEDV"], axis = 1) 
  y = df["MEDV"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

  #Model Creation
  lm = LinearRegression()
  lm.fit(X_train,y_train)

  #Model prediction
  Y_Pred = lm.predict(X_test)
  RMSE = np.sqrt(metrics.mean_squared_error(y_test, Y_Pred))

  signature = infer_signature(X_train, lm.predict(X_train))


  print(f"RMSE: {RMSE}")
  mlflow.log_metric("score", RMSE)

  mlflow.sklearn.log_model(lm, "model", signature=signature)
  print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

