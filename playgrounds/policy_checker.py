from tasks.code.hackathon_code.Utils.utils import *
from tasks.code.hackathon_code.Utils.pp import preprocess_q1



df = load_data("../task_1/code/hackathon_code/Datasets/train_set_agoda.csv")

df, y = preprocess_q1(df)

# df["test"] = df.apply(lambda row: parse_policy(row["cancellation_policy_code"], row["amount_nights"]), axis=1)




df[["days_cancellation_1",
    "percentage_cancellation_1",
    "days_cancellation_2",
    "percentage_cancellation_2",
    "no_show_percentage"]] = pd.DataFrame(df.apply(lambda row:
                                                   parse_policy(row["cancellation_policy_code"],
                                                                row["amount_nights"]), axis=1).tolist(), index=df.index)
print(df[["days_cancellation_1",
    "percentage_cancellation_1",
    "days_cancellation_2",
    "percentage_cancellation_2",
    "no_show_percentage"]].head(10))
