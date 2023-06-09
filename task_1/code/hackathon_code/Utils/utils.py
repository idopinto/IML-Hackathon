
from Base import baseline
import pandas as pd


def load_data(filename, cancel=True):
    if cancel:
        dates_to_parse = ["booking_datetime", "checkin_date",
                       "checkout_date", "hotel_live_date", "cancellation_datetime"]
    else:
        dates_to_parse = ["booking_datetime", "checkin_date",
                       "checkout_date", "hotel_live_date"]
    df = pd.read_csv(filename, parse_dates=dates_to_parse)
    return df
def calculate_percentage_and_count(df, feature):
    # Group by 'days_cancellation_1' column and apply lambda function
    # to calculate the percentage of 'True' values in 'label' column for each group,
    # as well as count the total number of labels in each group
    result = df.groupby(feature)['label'].agg([('y', lambda x: (x==True).mean()), ('count', 'count')])\
        .reset_index()

    # Rename the columns
    result.columns = ['x', 'y', 'count']

    return result

# def make_report(filename, title):
#     df = load_data(filename)
#     X, y = pp.preprocess_q1(df)
#     X['is_cancel'] = y
#     ProfileReport(X, title=title).to_file(title + ".html")
#     print("--------------------DONE--------------------")

def pipeline(X,y):
    # fit
    base = baseline.BaseEstimator().fit(X, y)
    # calculate loss
    f1 = base.loss(X, y)

    # Calculate precision and recall
    precision, recall = base.get_recall_precision(X, y)

    return f1, precision, recall


from datetime import datetime

def cancellation_cost(cancellation_policy,
                      cancellation_date,
                      checkin_date,
                      order_cost,
                      num_nights):
    cancellation_date = datetime.strptime(cancellation_date, "%Y-%m-%d")
    checkin_date = datetime.strptime(checkin_date, "%Y-%m-%d")
    days_until_checkin = (checkin_date - cancellation_date).days
    cost_per_night = order_cost / num_nights

    policies = cancellation_policy.split("_")
    no_show_policy = None

    to_return = 0

    for policy in policies:
        if "P" not in policy and "N" not in policy:
            raise ValueError("Invalid cancellation policy")
        if "D" not in policy:
            policy_type = policy[-1]  # Get the last character of policy, which is 'N' or 'P'
            charge = int(policy.split(policy_type)[0].split(' ')[-1])  # Get the charge value before 'N' or 'P'
            no_show_policy = (policy_type, charge)
            continue

        days, charge, policy_type = policy.split('D')[0], policy.split('D')[1][:-1], policy.split('D')[1][-1]
        days = int(days)
        charge = int(charge)

        if days_until_checkin <= days:
            if policy_type == 'N':
                to_return = charge * cost_per_night
            else:  # policy_type == 'P'
                to_return = (charge / 100) * order_cost

    if days_until_checkin <= 0 and no_show_policy is not None:
        policy_type, charge = no_show_policy
        if policy_type == 'N':
            return charge * cost_per_night
        else:  # policy_type == 'P'
            return (charge / 100) * order_cost

    return to_return



def parse_policy(policy_code, num_nights):
    policies = policy_code.split("_")
    parsed_policy = [-1, -1, -1, -1, -1]  # Initialize the list with -1

    if policy_code == "UNKNOWN": return parsed_policy

    for i, policy in enumerate(policies):
        if policy == '':
            continue
        if "D" not in policy:
            policy_type = policy[-1]  # Get the last character of policy, which is 'N' or 'P'
            charge = int(policy.split(policy_type)[0].split(' ')[-1])  # Get the charge value before 'N' or 'P'
            if policy_type == 'P':
                parsed_policy[4] = charge
            else:  # policy_type == 'N'
                parsed_policy[4] = (charge / num_nights) * 100
        else:
            days, charge, policy_type = policy.split('D')[0], policy.split('D')[1][:-1], policy.split('D')[1][-1]
            if policy_type == 'P':
                parsed_policy[i*2] = int(days)
                parsed_policy[i*2 + 1] = int(charge)
            else:  # policy_type == 'N'
                parsed_policy[i*2] = int(days)
                parsed_policy[i*2 + 1] = (int(charge) / num_nights) * 100

    return parsed_policy


# def run_XGBoost(X_train, y_train, X_test, y_test):
#     params = {
#         'max_depth': 7,
#         'learning_rate': 0.1,
#         'n_estimators': 300,
#         'gamma': 0.2,
#         'subsample': 1.0,
#         'colsample_bytree': 1.0,
#         'reg_alpha': 0.5,
#         'reg_lambda': 0
#     }
#     clf = XGBClassifier(**params)
#     clf.fit(X_train, y_train)
#
#     y_pred = clf.predict(X_test)
#     print_result("Test Results", y_test, y_pred)
#     # print(clf.feature_importances_)
#     # print(len(clf.feature_importances_))
#     # print(np.argmax(np.array(clf.feature_importances_)))
#     # print(X_train.columns[np.argmax(np.array(clf.feature_importances_))])
#     # Create a new DataFrame with 'id' and 'cancellation' columns
#     # df = pd.DataFrame({'id': id_column, 'cancellation': y_pred})
#
#     # Export the DataFrame to a CSV file
#     # df.to_csv('agoda_cancellation_prediction.csv', index=False)
#
#
# def run_lightGBM(clf, X_train, y_train, X_test, y_test, params):
#     randomized_search = RandomizedSearchCV(clf, params, scoring='f1', n_iter=10, cv=5)
#
#     # Fit the RandomizedSearchCV object to the training data
#     randomized_search.fit(X_train, y_train)
#
#     # Get the best parameters and the best F1 score
#     best_params = randomized_search.best_params_
#     best_score = randomized_search.best_score_
#     print("Train Results:")
#     print("\tbest params: ", best_params)
#     print("\tbest score: ", best_score)
#     # Make predictions on the test set using the best model
#     y_pred = randomized_search.predict(X_test)
#     y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]  # Convert probabilities to binary predictions
#     print_result("Test Results", y_test, y_pred_binary)


    # print(clf.feature_importances_)
    # print(len(clf.feature_importances_))
    # print(np.argmax(np.array(clf.feature_importances_)))
    # features = np.array(X_train.columns).reshape((-1, 1))
    # feature_and_importance = np.concatenate((features, np.array(clf.feature_importances_).reshape(-1, 1)), axis=1)
    # print(feature_and_importance[np.argsort(np.array(clf.feature_importances_))])

    # def evaluate_different_models_cv(X, y, classifiers, names, scoring):
    #     f1_results = []
    #     for i, classifier in enumerate(classifiers):
    #         # Perform cross-validation
    #         cv_results = cross_validate(classifier, X, y, scoring=scoring, cv=5)
    #         f1_results.append(cv_results['test_f1_macro'].mean())
    #         # Extract and print the mean scores
    #         print(f"{names[i]} Results")
    #
    #         # print("\tAccuracy:", cv_results['test_accuracy'].mean())
    #         # print("\tPrecision:", cv_results['test_precision_macro'].mean())
    #         # print("\tRecall:", cv_results['test_recall_macro'].mean())
    #         # print("\tF1 Score:", cv_results['test_f1_macro'].mean())
    #         # print("--------------------------------------------------")
    #     best_f1 = np.argmax(np.array(f1_results))
    #     print("The Best Model is:\n"
    #           f"\t {names[best_f1]}\n"
    #           f"\t f1 score: {f1_results[best_f1]}\n")
    #
    # def model_selection(models, names, params_grids, X, y, scoring='f1'):
    #     # GridSearchCV
    #     for i, model in enumerate(models):
    #         grid_search = GridSearchCV(model, params_grids[i], cv=5, scoring=scoring)
    #
    #         grid_search.fit(X, y)
    #         # Print the best parameters and the corresponding mean cross-validated score
    #
    #         print(f"GridSearchCV for {names[i]}- Best Parameters:", grid_search.best_params_)
    #         print(f"GridSearchCV for {names[i]} - Best F1 Score:", grid_search.best_score_)
    #         print("--------------------------------------------")
    #
    # def print_result(title, y_test, y_pred):
    #     print(title)
    #     print("\tF1: ", f1_score(y_test, y_pred))
    #     print("\tPrecision: ", precision_score(y_test, y_pred))
    #     print("\tRecall: ", recall_score(y_test, y_pred))


# X_test2, y_test2, z_test2 = pp.preprocess_q2(utils.load_data("hackathon_code/Datasets/test_set_agoda.csv"))
# y_pred2 = xgb2.predict(X_test2)
# y_pred3 = LR.predict(X_test2)
# y_pred3 = np.abs(y_pred3)
# y_pred3 = np.where(y_pred2 == 0, -1, y_pred3)
# # y_test2 = np.where(z_test2 == 0, -1, y_test2)
# #
# rmse = np.sqrt(mean_squared_error(y_test2, y_pred3))
# print("RMSE: ", rmse)
