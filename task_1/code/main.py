import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import utils, pp
# from hackathon_code.models import model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost as xgb

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, make_scorer


def evaluate_different_models_cv(X,y, classifiers,names, scoring):
    f1_results = []
    for i, classifier in enumerate(classifiers):
        # Perform cross-validation
        cv_results = cross_validate(classifier, X, y, scoring=scoring, cv=5)
        f1_results.append(cv_results['test_f1_macro'].mean())
        # Extract and print the mean scores
        print(f"{names[i]} Results")

        # print("\tAccuracy:", cv_results['test_accuracy'].mean())
        # print("\tPrecision:", cv_results['test_precision_macro'].mean())
        # print("\tRecall:", cv_results['test_recall_macro'].mean())
        # print("\tF1 Score:", cv_results['test_f1_macro'].mean())
        # print("--------------------------------------------------")
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

def find_best_params_xgboost(clf, X, y, param_grid, n_iter=10, cv=3):
    # # Split the data into train and test sets
    # Create the XGBoost classifier
    # xgb = XGBClassifier()

    # Create the RandomizedSearchCV object
    randomized_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                           n_iter=n_iter, cv=cv, scoring='f1')

    # Fit the RandomizedSearchCV object to the training data
    randomized_search.fit(X, y)

    # Get the best parameters and the best F1 score
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    # Make predictions on the test set using the best model
    y_pred = randomized_search.predict(X)

    # Calculate the F1 score on the test set
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y,y_pred)
    return randomized_search, best_params, best_score, f1, precision, recall

def print_result(title, y_test,y_pred):
    print(title)
    print("\tF1: ", f1_score(y_test, y_pred))
    print("\tPrecision: ", precision_score(y_test, y_pred))
    print("\tRecall: ", recall_score(y_test, y_pred))
def run_XGBoost(clf, params,X_train,y_train,X_test,y_test):

    # best_model, best_params, best_score, f1, precision, recall = find_best_params_xgboost(clf, X_train,y_train,params)
    randomized_search = RandomizedSearchCV(clf, param_distributions=params,n_iter=10, cv=5, scoring='f1')

    # Fit the RandomizedSearchCV object to the training data
    randomized_search.fit(X_train, y_train)

    # Get the best parameters and the best F1 score
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    # Make predictions on the test set using the best model
    y_pred = randomized_search.predict(X_test)

    print("Train Results:")
    print("\tbest params: ", best_params)
    print("\tbest score: ", best_score)
    # print("\tF1: ", f1_score(y_test, y_pred))
    # print("\tPrecision:" , precision_score(y_test, y_pred))
    # print("\tRecall: ",  recall_score(y_test,y_pred))
    print_result("Test Results", y_test,y_pred)

def run_lightGBM(clf,X_train,y_train,X_test,y_test,params):

    randomized_search = RandomizedSearchCV(clf, params, scoring='f1', n_iter=10, cv=5)

    # Fit the RandomizedSearchCV object to the training data
    randomized_search.fit(X_train, y_train)

    # Get the best parameters and the best F1 score
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    print("Train Results:")
    print("\tbest params: ", best_params)
    print("\tbest score: ", best_score)
    # Make predictions on the test set using the best model
    y_pred = randomized_search.predict(X_test)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]  # Convert probabilities to binary predictions
    print_result("Test Results", y_test, y_pred_binary)
def predict_cancellation():
    train_df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    test_df = utils.load_data("hackathon_code/Datasets/test_set_agoda.csv")
    X_train, y_train = pp.preprocess(train_df)
    X_test, y_test = pp.preprocess(test_df)

    param_grid1 = {
        'max_depth': [7],
        'learning_rate': [0.1],
        'n_estimators': [300],
        'gamma': [0.2],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'reg_alpha': [0.5],
        'reg_lambda': [0],
    }
    run_XGBoost(XGBClassifier(), param_grid1, X_train, y_train, X_test, y_test)

    param_grid2 = {
        'boosting_type': ['gbdt', 'dart', 'goss'],
        'num_leaves': [10, 20, 30, 40, 50],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5],
        'min_child_samples': [20, 50, 100]
    }
    run_lightGBM(lgb.LGBMClassifier(), X_train, y_train, X_test, y_test, param_grid2)

def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def predict_selling_amount():
    train_df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    test_df = utils.load_data("hackathon_code/Datasets/test_set_agoda.csv")
    features = ["amount_guests","amount_nights", "hotel_star_rating",
                "distance_booking_checkin", "booking_datetime", "hotel_country_code"]
    X_train, y_train = pp.preprocess(train_df)
    X_train, y_train = X_train[features], train_df['original_selling_amount']
    X_test, y_test = pp.preprocess(test_df)
    X_test, y_test = X_test[features], test_df['original_selling_amount']

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge, Lasso


    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print("LS RMSE: ", rmse)
    print("-----------------------------------------------------")
    cv_scores = cross_val_score(Ridge(alpha=0.01), X_train, y_train, cv=5, scoring=make_scorer(rmse_score))
    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores)
    print("RIDGE Mean CV score:", cv_scores.mean())

    cv_scores = cross_val_score(Lasso(alpha=0.01), X_train, y_train, cv=5, scoring=make_scorer(rmse_score))

    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores)
    print("LASSO Mean CV score:", cv_scores.mean())

if __name__ == '__main__':
    # predict_cancellation()
    predict_selling_amount()