import numpy as np
from matplotlib import pyplot as plt

from task_1.code.hackathon_code.Utils.utils import *
from task_1.code.hackathon_code.Utils.pp import preprocess_q1



def calculate_percentage_and_count(df):
    # Group by 'days_cancellation_1' column and apply lambda function
    # to calculate the percentage of 'True' values in 'label' column for each group,
    # as well as count the total number of labels in each group
    result = df.groupby('days_cancellation_1')['label'].agg([('y', lambda x: (x==True).mean()), ('count', 'count')]).reset_index()

    # Rename the columns
    result.columns = ['x', 'y', 'count']

    return result




df = load_data("../Datasets/train_set_agoda.csv")
df, y = preprocess_q1(df)
df["label"] = y
df = df[["days_cancellation_1", "label"]]




result = calculate_percentage_and_count(df)

result = result[result["count"] > 50]
# Line Plot
plt.figure(figsize=(10, 6))
plt.plot(result['x'], result['y'], marker='o', linestyle='-')
plt.xlabel('Group of days_cancellation_1')
plt.ylabel('Percentage of True labels')
plt.title('Percentage of True Labels by Group')
plt.savefig("fig1.png")
plt.show()






# Define the D range
D = np.linspace(0, 400, 1000)

# Define the function P
P = np.minimum(np.maximum(-D/3 + 105, 0), 100)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(D, P)
plt.title('A graph of the percentage of the cancellation cost in'
          ' relation to the days before check-in')
plt.xlabel('Days before check in')
plt.ylabel('Percentage of cancellation cost')
plt.grid(True)
plt.show()

