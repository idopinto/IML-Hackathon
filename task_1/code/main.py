import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from task_1.code.pp import preprocess

if __name__ == '__main__':
    filename = "Datasets/train_set_agoda.csv"

    df = pd.read_csv(filename,
                     parse_dates=["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date",
                                  "cancellation_datetime"])
    # df.head()
    X, y = preprocess(df)

    X.head()
    y.head()
