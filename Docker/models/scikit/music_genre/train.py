import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:9005")
mlflow.set_experiment("on-boarding-scikit")

with mlflow.start_run():
    print(mlflow.get_artifact_uri())
    # X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    # y = np.array([0, 0, 1, 1, 1, 0])
    # lr = LogisticRegression(random_state=0, solver='newton-cg')
    # lr.fit(X, y)
    # score = lr.score(X, y)
    df = pd.read_csv('./dataset/music_genre.csv', sep=',')
    df[df=='?'] = np.nan
    df = df.dropna()

    df.reset_index(inplace = True)
    df = df.drop(["index", "instance_id", "track_name", "artist_name"], axis=1)
    df.rename(columns = {'obtained_date':'time_signature'}, inplace = True)
    df['time_signature'] = df['time_signature'].replace('4-Apr', 4)
    df['time_signature'] = df['time_signature'].replace('3-Apr', 3)
    df['time_signature'] = df['time_signature'].replace('5-Apr', 5)
    df['time_signature'] = df['time_signature'].replace('1-Apr', 1)

    key_encoder = LabelEncoder()
    df['key'] = key_encoder.fit_transform(df['key'])

    mode_encoder = LabelEncoder()
    df['mode'] = mode_encoder.fit_transform(df['mode'])
    df['music_genre'] = df['music_genre'].replace({'Rap': 'Rap/Hip-Hop', 'Hip-Hop': 'Rap/Hip-Hop',
                                               'Jazz': 'Jazz/Blues', 'Blues': 'Jazz/Blues',
                                               'Anime': 'Electronic/Anime', 'Electronic': 'Electronic/Anime',
                                               'Rock': 'Rock/Country', 'Country': 'Rock/Country'})

    y = df['music_genre']
    X = df.drop('music_genre', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    #LogisticRegression
    #create a Logistic Regression Classifier
    log_reg = LogisticRegression(solver="liblinear", penalty="l1")

    #fit the classifier to the training data
    log_reg.fit(X_train,y_train)

    score_train = log_reg.score(X_train, y_train)
    score_test = log_reg.score(X_test, y_test)

    #print the accuracy on train and test data
    print("Accuracy on train data:", log_reg.score(X_train, y_train))
    print("Accuracy on test data:", log_reg.score(X_test, y_test))

    

    mlflow.log_metric("Train score", score_train)
    mlflow.log_metric("Test score", score_test)
    
    mlflow.sklearn.log_model(log_reg, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    mlflow.end_run()
