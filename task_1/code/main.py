from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import pp
from sklearn.metrics import precision_score, recall_score

if __name__ == '__main__':

    dates_to_parse = [["booking_datetime","checkin_date", "checkout_date", "hotel_live_date", "cancellation_datetime"]]
    df = pd.read_csv("hackathon_code/Datasets/train_set_agoda.csv",parse_dates=dates_to_parse)
    # preprocessing
    X, y = pp.preprocess(df)

    # fit
    model = baseline.BaseEstimator().fit(X, y)
    f1 = model.loss(y, model.predict(X))

    # Calculate precision
    precision = precision_score(y, model.predict(X))

    # Calculate recall
    recall = recall_score(y, model.predict(X))

    # Print the results
    print("Precision:", precision)
    print("Recall:", recall)