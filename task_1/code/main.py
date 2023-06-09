import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
import pandas as pd
from task_1.code.hackathon_code.Utils import utils, pp
from sklearn.ensemble import RandomForestRegressor
import sys

# Get the command-line arguments


if __name__ == '__main__':
    args = sys.argv

    # Block 1 - Load Data & Preprocessing
    train_df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    test_df = utils.load_data("hackathon_code/Datasets/test_set_agoda.csv")
    X_train, y_train = pp.preprocess_q1(train_df)  # y = cancellation date
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
    test_df1 = utils.load_data(
        "C:/Users/idop8/Desktop/My Files/PyCharmProjects/IML-Hackathon/AgodaCancellationChallenge/Agoda_Test_1.csv",
        cancel=False)

    test1_p = pp.preprocess_q1(test_df1, False)
    y_pred1 = xgb1.predict(test1_p)
    res1 = pd.DataFrame({'id': test_df1['h_booking_id'], 'cancellation': y_pred1})
    # .to_csv('../predictions/agoda_cancellation_prediction.csv', index=False)
    # ------------------------------------------------------------
    # task 2
    test_df2 = utils.load_data(
        "C:/Users/idop8/Desktop/My Files/PyCharmProjects/IML-Hackathon/AgodaCancellationChallenge/Agoda_Test_2.csv",
        cancel=False)

    test2_p = pp.preprocess_q2(test_df2, train=False)
    y_pred_cancel_date = xgb2.predict(test2_p)
    y_pred_sell_amount = RF.predict(test2_p)
    y_pred_sell_amount = np.abs(y_pred_sell_amount)
    y_pred_sell_amount = np.where(y_pred_cancel_date == 0, -1, y_pred_sell_amount)

    pd.DataFrame({'id': test_df2['h_booking_id'], 'predicted_selling_amount': y_pred_sell_amount}) \
        # .to_csv('../predictions/agoda_cost_of_cancellation.csv', index=False)
    # Block 3 - question 3 & 4 + plots in pdf'
    df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")

    df, y = pp.preprocess_q1(df)
    df['label'] = y
    df['distance_booking_checkin'] = df['distance_booking_checkin'].astype(int)

    for feature in ["distance_booking_checkin",
                    "no_of_adults",
                    "no_of_children",
                    "percentage_cancellation_1"]:
        result = utils.calculate_percentage_and_count(df, feature)
        result = result[result["count"] > 50]
        plt.figure(figsize=(10, 6))
        plt.plot(result['x'], result['y'], marker='o', linestyle='-')
        plt.xlabel('Group of ' + feature)
        plt.ylabel('Percentage of True labels')
        plt.title(feature)
        plt.savefig(feature+"_fig.png")

    result = utils.calculate_percentage_and_count(df, "pay_now")
    result = result[result["count"] > 50]
    plt.figure(figsize=(10, 6))
    plt.bar(result['x'], result['y'])
    plt.xlabel('Group of ' + "pay_now")
    plt.ylabel('Percentage of True labels')
    plt.title("pay_now")
    plt.savefig("pay_now_fig.png")

