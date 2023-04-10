# Importing dependencies
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import sys

# Setting an experiment for automatic logging
mlflow.set_tracking_uri("http://localhost:9005")
mlflow.set_experiment(experiment_name="auto_log")

# Function for automatic logging
if __name__ == "__main__":   
    # Enabling automatic logging for scikit-learn runs
    mlflow.sklearn.autolog()
    
    # Loading data
    data = datasets.load_breast_cancer()
    
    # Setting hyperparameter values to try
    params = {"C": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    
    # Instantiating LogisticRegression and GridSearchCV
    log_reg = LogisticRegression(max_iter=int(sys.argv[1]))
    grid_search = GridSearchCV(log_reg, params)
    
    # Starting a logging run
    with mlflow.start_run() as run:
        # Fitting GridSearchCV
        grid_search.fit(data["data"], data["target"])
            
    # Disabling autologging
    mlflow.sklearn.autolog(disable=True)