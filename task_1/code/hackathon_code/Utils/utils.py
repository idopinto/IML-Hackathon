from sklearn.ensemble import RandomForestClassifier
from ydata_profiling import ProfileReport

from ..Base import baseline
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


def load_data(filename):
    dates_to_parse = ["booking_datetime", "checkin_date",
                       "checkout_date", "hotel_live_date", "cancellation_datetime"]
    df = pd.read_csv(filename, parse_dates=dates_to_parse)
    return df

def make_report(filename, title):
    df = load_data(filename)
    X, y = preprocess(df)
    X['is_cancel'] = y
    ProfileReport(X, title=title).to_file(title + ".html")
    print("--------------------DONE--------------------")

def pipeline(X,y):
    # fit
    base = baseline.BaseEstimator().fit(X, y)
    # calculate loss
    f1 = base.loss(X, y)

    # Calculate precision and recall
    precision, recall = base.get_recall_precision(X, y)

    return f1, precision, recall


from datetime import datetime

def cancellation_cost(cancellation_policy,
                      cancellation_date,
                      checkin_date,
                      order_cost,
                      num_nights):
    cancellation_date = datetime.strptime(cancellation_date, "%Y-%m-%d")
    checkin_date = datetime.strptime(checkin_date, "%Y-%m-%d")
    days_until_checkin = (checkin_date - cancellation_date).days
    cost_per_night = order_cost / num_nights

    policies = cancellation_policy.split("_")
    no_show_policy = None

    to_return = 0

    for policy in policies:
        if "P" not in policy and "N" not in policy:
            raise ValueError("Invalid cancellation policy")
        if "D" not in policy:
            policy_type = policy[-1]  # Get the last character of policy, which is 'N' or 'P'
            charge = int(policy.split(policy_type)[0].split(' ')[-1])  # Get the charge value before 'N' or 'P'
            no_show_policy = (policy_type, charge)
            continue

        days, charge, policy_type = policy.split('D')[0], policy.split('D')[1][:-1], policy.split('D')[1][-1]
        days = int(days)
        charge = int(charge)

        if days_until_checkin <= days:
            if policy_type == 'N':
                to_return = charge * cost_per_night
            else:  # policy_type == 'P'
                to_return = (charge / 100) * order_cost

    if days_until_checkin <= 0 and no_show_policy is not None:
        policy_type, charge = no_show_policy
        if policy_type == 'N':
            return charge * cost_per_night
        else:  # policy_type == 'P'
            return (charge / 100) * order_cost

    return to_return



def parse_policy(policy_code, num_nights):
    policies = policy_code.split("_")
    parsed_policy = [-1, -1, -1, -1, -1]  # Initialize the list with -1

    if policy_code == "UNKNOWN": return parsed_policy

    for i, policy in enumerate(policies):
        if policy == '':
            continue
        if "D" not in policy:
            policy_type = policy[-1]  # Get the last character of policy, which is 'N' or 'P'
            charge = int(policy.split(policy_type)[0].split(' ')[-1])  # Get the charge value before 'N' or 'P'
            if policy_type == 'P':
                parsed_policy[4] = charge
            else:  # policy_type == 'N'
                parsed_policy[4] = (charge / num_nights) * 100
        else:
            days, charge, policy_type = policy.split('D')[0], policy.split('D')[1][:-1], policy.split('D')[1][-1]
            if policy_type == 'P':
                parsed_policy[i*2] = int(days)
                parsed_policy[i*2 + 1] = int(charge)
            else:  # policy_type == 'N'
                parsed_policy[i*2] = int(days)
                parsed_policy[i*2 + 1] = (int(charge) / num_nights) * 100

    return parsed_policy


