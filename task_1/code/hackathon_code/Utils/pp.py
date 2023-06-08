import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    encoder = LabelEncoder()
    df['language'] = encoder.fit_transform(df['language'])
    df['hotel_country_code'] = encoder.fit_transform(df['hotel_country_code'])
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"])
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"])
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"])


    df["did_cancel"] = ~df["cancellation_datetime"].isna()
    df["distance_booking_checkin"] = ((df["checkin_date"] - df["booking_datetime"]) / pd.Timedelta(days=1)).astype(int)
    df["amount_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["amount_nights"] = ((df["checkout_date"] - df["checkin_date"]) / pd.Timedelta(days=1)).astype(int)
    df["checkin_date"] = df["checkin_date"].dt.dayofyear
    df["checkout_date"] = df["checkout_date"].dt.dayofyear
    df["booking_datetime"] = df["booking_datetime"].dt.dayofyear
    df["price_per_guest_per_night"] = df["original_selling_amount"] / (df["amount_guests"] * df["amount_nights"])
    df["costumer_guest_same_nation"] = df["customer_nationality"] == df["guest_nationality_country_name"]
    df["pay_now"] = df["charge_option"] == "Pay Now"
    y = df["did_cancel"]
    df = df.drop(["h_booking_id", "did_cancel", "h_customer_id", "request_nonesmoke", "request_airport"], axis=1)
    return df, y

