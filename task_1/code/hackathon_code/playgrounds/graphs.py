import numpy as np
from matplotlib import pyplot as plt

from task_1.code.hackathon_code.Utils.utils import *
from task_1.code.hackathon_code.Utils.pp import preprocess_q1

def calculate_percentage_and_count(df, feacher):
    # Group by 'days_cancellation_1' column and apply lambda function
    # to calculate the percentage of 'True' values in 'label' column for each group,
    # as well as count the total number of labels in each group
    result = df.groupby(feacher)['label'].agg([('y', lambda x: (x==True).mean()), ('count', 'count')]).reset_index()

    # Rename the columns
    result.columns = ['x', 'y', 'count']

    return result


df = load_data("../Datasets/train_set_agoda.csv")
df, y = preprocess_q1(df)
df["label"] = y


def calculate_cancellation_percent(df):
    # Select rows where has_children is True
    bookings_with_children = df[df['no_of_adults'] > 4]

    # Calculate percentage of cancellations
    child_cancellation_percent = bookings_with_children['label'].mean()

    bookings_without_children = df[df['no_of_adults'] <= 4]

    # Calculate percentage of cancellations
    nochild_cancellation_percent = bookings_without_children['label'].mean()

    return child_cancellation_percent, nochild_cancellation_percent

# print(df.columns)
print(calculate_cancellation_percent(df))
for feature in ["pay_now", "distance_booking_checkin", "no_of_children", "no_of_adults"]:
    if feature == "label": continue
    data = df[[feature, "label"]].astype(int)
    result = calculate_percentage_and_count(data, feature)

    result = result[result["count"] > 50]
    if np.max(result["y"]) - np.min(result["y"]) < 0.2: continue
    # Line Plot
    plt.figure(figsize=(10, 6))
    plt.plot(result['x'], result['y'], marker='o', linestyle='-')
    plt.xlabel('Group of '+feature)
    plt.ylabel('Percentage of True labels')
    plt.title(feature)
    # plt.savefig("fig1.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(result['x'], result['y'])
    plt.xlabel('Group of ' + feature)
    plt.ylabel('Percentage of True labels')
    plt.title(feature)
    # plt.savefig("fig1.png")
    plt.show()