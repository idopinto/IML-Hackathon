from sklearn.neighbors import KNeighborsClassifier

from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import utils, pp
from hackathon_code.models import model


import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score









def run_baseline(X,y):
    f1, precision, recall = utils.pipeline(X, y)

    print("Baseline model Results :\n"
          f"\t Precision: {precision}\n"
          f"\t Recall: {recall}\n"
          f"\t F1: {f1}\n"
          f"--------------------------------------\n")

if __name__ == '__main__':
    df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    features = ['pay_now', 'hotel_id', 'language', 'hotel_star_rating']
    # preprocessing
    X, y = pp.preprocess(df)
    X = X[features]
    run_baseline(X, y)
    models = [LogisticRegression(),
              KNeighborsClassifier(n_neighbors=5),
              RandomForestClassifier(n_estimators=100, random_state=42)]
    names = ["Logistic Regression", "KNN", "Random Forest"]
    f1_scores = []
    for i, model in enumerate(models):
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1_scores.append(f1_score(y, y_pred))
        print(f"{names[i]} Results")
        print("\tAccuracy: ", accuracy)
        print("\tF1 Score: ", f1_scores[i])
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
