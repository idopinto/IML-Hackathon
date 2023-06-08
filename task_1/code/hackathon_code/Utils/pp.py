import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from task_1.code.hackathon_code.Utils.utils import parse_policy


def preprocess_q1(df):
    encoder = LabelEncoder()
    df['language'] = encoder.fit_transform(df['language']).astype(float)
    df['hotel_country_code'] = encoder.fit_transform(df['hotel_country_code']).astype(float)
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"]).astype(float)

    df['request_nonesmoke'] = df['request_nonesmoke'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_airport'] = df['request_airport'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_latecheckin'] = df['request_latecheckin'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_highfloor'] = df['request_highfloor'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_twinbeds'] = df['request_twinbeds'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_earlycheckin'] = df['request_earlycheckin'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_largebed'] = df['request_largebed'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['hotel_brand_code'] = df['hotel_brand_code'].apply(lambda x: x if x >= 0 else -1)
    df['hotel_chain_code'] = df['hotel_chain_code'].apply(lambda x: x if x >= 0 else -1)

    df[["days_cancellation_1",
        "percentage_cancellation_1",
        "days_cancellation_2",
        "percentage_cancellation_2",
        "no_show_percentage"]] = pd.DataFrame(df.apply(lambda row:
                                                       parse_policy(row["cancellation_policy_code"],
                                                                    row["amount_nights"]), axis=1).tolist(),
                                              index=df.index)
    # df['cancellation_policy_code'] = df['cancellation_policy_code'].apply(
    #     lambda x: re.findall(r'(\d+)D', x)[0] if re.findall(r'(\d+)D', x) else 0).astype(float)

    df["has_request"] = (df['request_nonesmoke'] + df['request_airport'] + df['request_latecheckin'] +
                         df['request_highfloor'] + df['request_twinbeds'] + df['request_earlycheckin'] +
                         df['request_largebed'])

    df["did_cancel"] = ~df["cancellation_datetime"].isna()
    df["distance_booking_checkin"] = ((df["checkin_date"] - df["booking_datetime"]) / pd.Timedelta(days=1)).astype(
        float)
    df["amount_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["amount_nights"] = ((df["checkout_date"] - df["checkin_date"]) / pd.Timedelta(days=1)).astype(float)
    df["hotel_live_date"] = ((df["checkin_date"] - df["hotel_live_date"]) / pd.Timedelta(days=1)).astype(float)
    df["checkin_date"] = df["checkin_date"].dt.dayofyear
    df["checkout_date"] = df["checkout_date"].dt.dayofyear
    df["booking_datetime"] = df["booking_datetime"].dt.dayofyear
    df["price_per_guest_per_night"] = df["original_selling_amount"] / (df["amount_guests"] * df["amount_nights"])
    df["costumer_guest_same_nation"] = (df["customer_nationality"] == df["guest_nationality_country_name"]) == \
                                       (df["customer_nationality"] == df['origin_country_code'])

    df['origin_country_code'] = encoder.fit_transform(df['origin_country_code']).astype(float)
    df['guest_nationality_country_name'] = encoder.fit_transform(df['guest_nationality_country_name']).astype(float)
    df['customer_nationality'] = encoder.fit_transform(df['customer_nationality']).astype(float)
    df['original_payment_method'] = encoder.fit_transform(df['original_payment_method']).astype(float)
    df['original_payment_type'] = encoder.fit_transform(df['original_payment_type']).astype(float)
    df['original_payment_currency'] = encoder.fit_transform(df['original_payment_currency']).astype(float)

    df["pay_now"] = df["charge_option"] == "Pay Now"
    y = df["did_cancel"]
    df = df.drop(["did_cancel", "h_customer_id", "cancellation_datetime",
                  "hotel_brand_code", "hotel_chain_code", "charge_option"], axis=1)
    return df, y


def preprocess_q2(df):
    encoder = LabelEncoder()
    df['language'] = encoder.fit_transform(df['language']).astype(float)
    df['hotel_country_code'] = encoder.fit_transform(df['hotel_country_code']).astype(float)
    df["accommadation_type_name"] = encoder.fit_transform(df["accommadation_type_name"]).astype(float)

    df['request_nonesmoke'] = df['request_nonesmoke'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_airport'] = df['request_airport'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_latecheckin'] = df['request_latecheckin'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_highfloor'] = df['request_highfloor'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_twinbeds'] = df['request_twinbeds'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_earlycheckin'] = df['request_earlycheckin'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['request_largebed'] = df['request_largebed'].apply(lambda x: x if (x == 0) or (x == 1) else 0)
    df['hotel_brand_code'] = df['hotel_brand_code'].apply(lambda x: x if x >= 0 else -1)
    df['hotel_chain_code'] = df['hotel_chain_code'].apply(lambda x: x if x >= 0 else -1)
    # df["original_selling_amount"] = df["original_selling_amount"] * (df["did_cancel"].astype(int))
    # df['original_selling_amount'] = df['original_selling_amount'].apply(lambda x: x if x > 0 else -1)

    # df['cancellation_policy_code'] = df['cancellation_policy_code'].apply(
    #     lambda x: re.findall(r'(\d+)D', x)[0] if re.findall(r'(\d+)D', x) else 0).astype(float)

    df["has_request"] = (df['request_nonesmoke'] + df['request_airport'] + df['request_latecheckin'] +
                         df['request_highfloor'] + df['request_twinbeds'] + df['request_earlycheckin'] +
                         df['request_largebed'])

    df["did_cancel"] = ~df["cancellation_datetime"].isna()
    df["distance_booking_checkin"] = ((df["checkin_date"] - df["booking_datetime"]) / pd.Timedelta(days=1)).astype(
        float)
    df["amount_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["amount_nights"] = ((df["checkout_date"] - df["checkin_date"]) / pd.Timedelta(days=1)).astype(float)
    df["hotel_live_date"] = ((df["checkin_date"] - df["hotel_live_date"]) / pd.Timedelta(days=1)).astype(float)
    df["checkin_date"] = df["checkin_date"].dt.dayofyear
    df["checkout_date"] = df["checkout_date"].dt.dayofyear
    df["booking_datetime"] = df["booking_datetime"].dt.dayofyear

    # df["t"] = [df["cancellation_policy_code"],df["amount_nights"]]
    # df["days_cancellation_1"] = df.apply(
    #     lambda row: parse_policy(row["cancellation_policy_code"], row["amount_nights"])[0], axis=1)
    # df["days_cancellation_1"] = parse_policy(df["cancellation_policy_code"], df["amount_nights"])
    # df["days_cancellation_1"] = df['cancellation_policy_code'].apply(lambda x: parse_policy(x) if True else -1)

    df[["days_cancellation_1",
        "percentage_cancellation_1",
        "days_cancellation_2",
        "percentage_cancellation_2",
        "no_show_percentage"]] = pd.DataFrame(df.apply(lambda row:
                                                       parse_policy(row["cancellation_policy_code"],
                                                                    row["amount_nights"]), axis=1).tolist(),index=df.index)

    df["costumer_guest_same_nation"] = (df["customer_nationality"] == df["guest_nationality_country_name"]) == \
                                       (df["customer_nationality"] == df['origin_country_code'])

    df['origin_country_code'] = encoder.fit_transform(df['origin_country_code']).astype(float)
    df['guest_nationality_country_name'] = encoder.fit_transform(df['guest_nationality_country_name']).astype(float)
    df['customer_nationality'] = encoder.fit_transform(df['customer_nationality']).astype(float)
    df['original_payment_method'] = encoder.fit_transform(df['original_payment_method']).astype(float)
    df['original_payment_type'] = encoder.fit_transform(df['original_payment_type']).astype(float)
    df['original_payment_currency'] = encoder.fit_transform(df['original_payment_currency']).astype(float)

    df["pay_now"] = df["charge_option"] == "Pay Now"
    y = df["original_selling_amount"]
    df = df.drop(["did_cancel", "h_customer_id", "cancellation_datetime",
                  "hotel_brand_code", "hotel_chain_code", "charge_option", "original_selling_amount"], axis=1)
    return df, y
