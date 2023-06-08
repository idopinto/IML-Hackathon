import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import utils, pp
# from hackathon_code.models import model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score







def run_lightGBN(X, y):
    # # Prepare the dataset for LightGBM
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

def evaluate_different_models_cv(classifiers, names, scoring):
    f1_results = []
    for i, classifier in enumerate(classifiers):
        # classifier.fit(X, y)
        # Perform cross-validation
        cv_results = cross_validate(classifier, X, y, scoring=scoring, cv=5)
        f1_results.append(cv_results['test_f1_macro'].mean())
        # Extract and print the mean scores
        print(f"{names[i]} Results")

        print("\tAccuracy:", cv_results['test_accuracy'].mean())
        print("\tPrecision:", cv_results['test_precision_macro'].mean())
        print("\tRecall:", cv_results['test_recall_macro'].mean())
        print("\tF1 Score:", cv_results['test_f1_macro'].mean())
        print("--------------------------------------------------")
    best_f1 = np.argmax(np.array(f1_results))
    print("The Best Model is:\n"
          f"\t {names[best_f1]}\n"
          f"\t f1 score: {f1_results[best_f1]}\n")


def run_baseline(X,y):
    f1, precision, recall = utils.pipeline(X, y)

    print("Baseline model Results :\n"
          f"\t Precision: {precision}\n"
          f"\t Recall: {recall}\n"
          f"\t F1: {f1}\n"
          f"--------------------------------------\n")

def model_selection(models,names,params_grids, X,y,scoring='f1'):
    # GridSearchCV
    for i, model in enumerate(models):
        grid_search = GridSearchCV(model, params_grids[i], cv=5, scoring=scoring)

        grid_search.fit(X, y)
        # Print the best parameters and the corresponding mean cross-validated score

        print(f"GridSearchCV for {names[i]}- Best Parameters:", grid_search.best_params_)
        print(f"GridSearchCV for {names[i]} - Best F1 Score:", grid_search.best_score_)
        print("--------------------------------------------")


if __name__ == '__main__':
    df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    # features = ['pay_now', 'hotel_id', 'language', 'hotel_star_rating',"distance_booking_checkin"]

    # preprocessing
    X, y = pp.preprocess(df)
    features = X.select_dtypes(include=['float64', 'int64','boolean']).columns
    X = X[features]
    # print(X.columns)
    # print(X.shape)
    # run_baseline(X, y)
    # # run_lightGBN(X, y)
    # ------------------------------
    classifiers = [LogisticRegression(),
              KNeighborsClassifier(),
              RandomForestClassifier()]
    names = ["Logistic Regression" ,"KNN", "Random Forest"]
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    # evaluate_different_models_cv(classfiers,names,scoring)

    logreg_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    rf_grid = {
        'n_estimators': [50, 100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced'],
        'max_samples': [None, 0.5, 0.8]
    }
    params = [logreg_param_grid,knn_param_grid, rf_grid]
    model_selection(classifiers,names, params, X, y)
