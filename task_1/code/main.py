from sklearn.neighbors import KNeighborsClassifier

from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import utils, pp
from sklearn.metrics import precision_score, recall_score, f1_score
from hackathon_code.models import model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

if __name__ == '__main__':
    df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    features = ['pay_now', 'hotel_id', 'language', 'hotel_star_rating']
    # preprocessing
    X, y = pp.preprocess(df)
    X = X[features]
    f1, precision, recall = utils.pipeline(X, y)

    print("Baseline model Results :\n"
          f"\t Precision: {precision}\n"
          f"\t Recall: {recall}\n"
          f"\t F1: {f1}\n"
          f"--------------------------------------\n")

    # Initialize the logistic regression model

    logreg = LogisticRegression()
    # Train the model on the training data
    logreg.fit(X, y)

    # Make predictions on the test data
    y_pred = logreg.predict(X)
    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y, y_pred)
    print("Logistic Regression Results")
    print("\tAccuracy: ", accuracy)
    print("\tF1 Score: ", f1_score(y, y_pred))

    knn = KNeighborsClassifier(n_neighbors=5)
# Train the classifier on the training data
    knn.fit(X, y)

    # Make predictions on the test data
    y_pred = knn.predict(X)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y, y_pred)
    print("KNN Results:")
    print("\tKNN Accuracy:", accuracy)
    print("\tKNN F1 Score: ", f1_score(y, y_pred))
    print("--------------------------------------")
    # Prepare the dataset for LightGBM
    train_data = lgb.Dataset(X, label=y)

    # Set the parameters for LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }


    # Train the LightGBM model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Make predictions on the test data
    y_pred = model.predict(X)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Convert probabilities to class labels

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y, y_pred)
    print("LightGBM Results")
    print("\tAccuracy:", accuracy)
    print("\tF1 Score: ", f1_score(y, y_pred))
