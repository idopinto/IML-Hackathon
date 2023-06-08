import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from task_1.code.pp import preprocess
from sklearn.preprocessing import LabelEncoder


def preprocess(df):
    encoder = LabelEncoder()

    # encode language, hotel_country_code, accommadation_type_name
    df['language'] = encoder.fit_transform(df['language'])
    df['hotel_country_code'] = encoder.fit_transform(df['hotel_country_code'])
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"])

    # create new feature - boolean did cancel?
    df["did_cancel"] = ~df["cancellation_datetime"].isna()

    # create new feature - days between booking and check in
    df["distance_booking_checkin"] = ((df["checkin_date"] - df["booking_datetime"]) / pd.Timedelta(days=1)).astype(int)

    # create new feature - amount of guests = adults + children
    df["amount_guests"] = df["no_of_adults"] + df["no_of_children"]

    # create new feature - for how many days the booking
    df["amount_nights"] = ((df["checkout_date"] - df["checkin_date"]) / pd.Timedelta(days=1)).astype(int)

    # change checkin_date, checkout_date, booking_datetime to be only day of year (disregard year)
    df["checkin_date"] = df["checkin_date"].dt.dayofyear
    df["checkout_date"] = df["checkout_date"].dt.dayofyear
    df["booking_datetime"] = df["booking_datetime"].dt.dayofyear

    # # create new feature - price_per_guest_per_night
    df["price_per_guest_per_night"] = df["original_selling_amount"] / (df["amount_guests"] * df["amount_nights"])

    # create new feature - true if the guest and the customer from the same nation
    df["costumer_guest_same_nation"] = df["customer_nationality"] == df["guest_nationality_country_name"]

    # create new feature - true if the customer pays when booking, false if he pays later
    df["pay_now"] = df["charge_option"] == "Pay Now"

    #
    y = df["did_cancel"]
    df = df.drop(["h_booking_id", "did_cancel", "h_customer_id", "request_nonesmoke", "request_airport"], axis=1)
    return df, y


if __name__ == '__main__':
    filename = "../Datasets/train_set_agoda.csv"

    df = pd.read_csv(filename,
                     parse_dates=["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date",
                                  "cancellation_datetime"])
    # df.head()
    X, y = preprocess(df)

    # X2 = X[df["hotel_country_code"].isna()]
    # X2 = X2[X2["amount_nights"] > 1]
    # pd.set_option('display.max_columns', None)
    # print(X["distance_booking_checkin"] < 0)
    # print(y)
    # print(X2[["hotel_country_code", "hotel_id", "hotel_city_code"]])
    print(X['language'])

