import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv(
        "C:\\Users\\matan\\PycharmProjects\\IML-Hackathon\\AgodaCancellationChallenge\\agoda_cancellation_train.csv")
    df["did_cancel"] = df["cancellation_datetime"].isna()
    df["amount_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["price_per_guest_per_night"] = df["original_selling_amount"] / (df["amount_guests"] * df["amount_nights"])
    df["costumer_guest_same_nation"] = df["customer_nationality"] == df["guest_nationality_country_name"]
    df["pay_now"] = df["charge_option"] == "pay_now"
    df["distance_booking_checkin"]
    df["amount_nights"]
