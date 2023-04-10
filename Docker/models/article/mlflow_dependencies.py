import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets


mlflow.set_tracking_uri("http://localhost:9005")