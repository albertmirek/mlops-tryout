import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from mlflow.models import infer_signature, ModelSignature
from mlflow.types import Schema, ColSpec


# Loading data
data = datasets.load_breast_cancer()
    
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                    data.target,
                                                    stratify=data.target)

# Instantiating and fitting the model
model = LogisticRegression(max_iter=1000)            
model.fit(X=X_train, y=y_train)


# Converting train features into a DataFrame
X_train_df = pd.DataFrame(data=X_train, columns=data.feature_names)

# Inferring the input signature
signature = infer_signature(model_input=X_train_df, 
                            model_output=model.predict(X_test))

# Example input schema for the Iris dataset
# input_schema = Schema(inputs=[
#     ColSpec(type="double", name="sepal length (cm)"),
#     ColSpec(type="double", name="sepal width (cm)"),
#     ColSpec(type="double", name="petal length (cm)"),
#     ColSpec(type="double", name="petal width (cm)"),
# ])

# # Example input schema for the Iris dataset
# output_schema = Schema(inputs=[ColSpec(type="long")])

# # Creating an input schema for the breast cancer dataset
# input_schema = Schema(inputs=[ColSpec(type="double", name=feature_name) 
#                               for feature_name in data.feature_names])

# # Creating an output schema for the breast cancer dataset
# output_schema = Schema(inputs=[ColSpec("long")])  

# # Creatubg a signature from our schemas
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# # Creating an input example from our feature DataFrame
# input_example = X_train_df.iloc[:2]

# Specifying a conda environment
conda_env = {
    "channels": ["default"],
    "dependencies": ["pip"],
    "pip": ["mlflow", "cloudpickle==1.6.0"],
    "name": "mlflow-env"}

# Specifying pip requirements
pip_requirements = ["mlflow"]

# Saving the model 

mlflow.sklearn.save_model(sk_model=model, 
                          path="model", 
                          conda_env=conda_env, 
                          signature=signature,
                          input_example=input_example)

# Saving the model as an artifact in a run
with mlflow.start_run() as run:
    # Obtaining the ID of this run
    run_id = run.info.run_id
    
    # Logging our model
    mlflow.sklearn.log_model(sk_model=model, 
                             artifact_path="model", 
                             conda_env=conda_env, 
                             signature=signature,
                             input_example=input_example)


# Path to the model saved with log_model
model_uri_logged = "runs:/{run_id}/model"

# Path to the model saved with save_model
model_uri_saved = "model"

# Loading our model as a Python function
pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_uri_saved)

# Loading our model as a scikit-learn model
sklearn_model = mlflow.sklearn.load_model(model_uri=model_uri_saved)