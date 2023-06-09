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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor


# X_test2, y_test2, z_test2 = pp.preprocess_q2(utils.load_data("hackathon_code/Datasets/test_set_agoda.csv"))
# y_pred2 = xgb2.predict(X_test2)
# y_pred3 = LR.predict(X_test2)
# y_pred3 = np.abs(y_pred3)
# y_pred3 = np.where(y_pred2 == 0, -1, y_pred3)
# # y_test2 = np.where(z_test2 == 0, -1, y_test2)
# #
# rmse = np.sqrt(mean_squared_error(y_test2, y_pred3))
# print("RMSE: ", rmse)
def evaluate_different_models_cv(X, y, classifiers, names, scoring):
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


def run_baseline(X, y):
    f1, precision, recall = utils.pipeline(X, y)

    print("Baseline model Results :\n"
          f"\t Precision: {precision}\n"
          f"\t Recall: {recall}\n"
          f"\t F1: {f1}\n"
          f"--------------------------------------\n")


def model_selection(models, names, params_grids, X, y, scoring='f1'):
    # GridSearchCV
    for i, model in enumerate(models):
        grid_search = GridSearchCV(model, params_grids[i], cv=5, scoring=scoring)

        grid_search.fit(X, y)
        # Print the best parameters and the corresponding mean cross-validated score

        print(f"GridSearchCV for {names[i]}- Best Parameters:", grid_search.best_params_)
        print(f"GridSearchCV for {names[i]} - Best F1 Score:", grid_search.best_score_)
        print("--------------------------------------------")


def find_best_params_xgboost(clf, X, y, param_grid, n_iter=10, cv=5):
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
    recall = recall_score(y, y_pred)
    return randomized_search, best_params, best_score, f1, precision, recall


def print_result(title, y_test, y_pred):
    print(title)
    print("\tF1: ", f1_score(y_test, y_pred))
    print("\tPrecision: ", precision_score(y_test, y_pred))
    print("\tRecall: ", recall_score(y_test, y_pred))


def run_XGBoost(X_train, y_train, X_test, y_test):
    params = {
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'gamma': 0.2,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.5,
        'reg_lambda': 0
    }
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print_result("Test Results", y_test, y_pred)
    # print(clf.feature_importances_)
    # print(len(clf.feature_importances_))
    # print(np.argmax(np.array(clf.feature_importances_)))
    # print(X_train.columns[np.argmax(np.array(clf.feature_importances_))])
    # Create a new DataFrame with 'id' and 'cancellation' columns
    # df = pd.DataFrame({'id': id_column, 'cancellation': y_pred})

    # Export the DataFrame to a CSV file
    # df.to_csv('agoda_cancellation_prediction.csv', index=False)


def run_lightGBM(clf, X_train, y_train, X_test, y_test, params):
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



    # print(clf.feature_importances_)
    # print(len(clf.feature_importances_))
    # print(np.argmax(np.array(clf.feature_importances_)))
    # features = np.array(X_train.columns).reshape((-1, 1))
    # feature_and_importance = np.concatenate((features, np.array(clf.feature_importances_).reshape(-1, 1)), axis=1)
    # print(feature_and_importance[np.argsort(np.array(clf.feature_importances_))])

if __name__ == '__main__':
    # Block 1 - Load Data & Preprocessing
    train_df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    test_df = utils.load_data("hackathon_code/Datasets/test_set_agoda.csv")
    X_train, y_train = pp.preprocess_q1(train_df) # y = cancellation date
    X_test, y_test = pp.preprocess_q1(test_df)

    params = {
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'gamma': 0.2,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.5,
        'reg_lambda': 0
    }

    # train XGBoost for task 1
    xgb1 = XGBClassifier(**params)
    xgb1.fit(X_train, y_train)

    # preprocess for task 2
    X_train2, y_train2, z_train2 = pp.preprocess_q2(utils.load_data("hackathon_code/Datasets/train_set_agoda.csv"))

    # train XGBoost for task 2
    xgb2 = XGBClassifier(**params)
    xgb2.fit(X_train2, z_train2)

    RF = RandomForestRegressor()
    RF.fit(X_train2, y_train2)

    # Block 2 - Loading Test Sets for each task

    # Task 1
    test_df1 = utils.load_data("C:/Users/idop8/Desktop/My Files/PyCharmProjects/IML-Hackathon/AgodaCancellationChallenge/Agoda_Test_1.csv",
                               cancel=False)

    test1_p = pp.preprocess_q1(test_df1, False)
    y_pred1 = xgb1.predict(test1_p)
    res1 = pd.DataFrame({'id': test_df1['h_booking_id'], 'cancellation': y_pred1})
        # .to_csv('agoda_cancellation_prediction.csv', index=False)
    # ------------------------------------------------------------
    # task 2
    test_df2 = utils.load_data("C:/Users/idop8/Desktop/My Files/PyCharmProjects/IML-Hackathon/AgodaCancellationChallenge/Agoda_Test_2.csv",
                               cancel= False)

    test2_p = pp.preprocess_q2(test_df2, train=False)
    y_pred_cancel_date = xgb2.predict(test2_p)
    y_pred_sell_amount = RF.predict(test2_p)
    y_pred_sell_amount = np.abs(y_pred_sell_amount)
    y_pred_sell_amount = np.where(y_pred_cancel_date == 0, -1, y_pred_sell_amount)

    pd.DataFrame({'id': test_df2['h_booking_id'], 'predicted_selling_amount': y_pred_sell_amount}) \
        .to_csv('../predictions/agoda_cost_of_cancellation.csv', index=False)
    # Block 3 - question 3 & 4 + plots in pdf'

