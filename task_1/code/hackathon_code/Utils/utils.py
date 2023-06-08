from sklearn.ensemble import RandomForestClassifier

from . import pp
from ..Base import baseline
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


def load_data(filename):
    dates_to_parse = ["booking_datetime", "checkin_date",
                       "checkout_date", "hotel_live_date", "cancellation_datetime"]
    df = pd.read_csv(filename, parse_dates=dates_to_parse)
    return df
def pipeline(X,y):
    # fit
    base = baseline.BaseEstimator().fit(X, y)
    # calculate loss
    f1 = base.loss(X, y)

    # Calculate precision and recall
    precision, recall = base.get_recall_precision(X, y)

    return f1, precision, recall
