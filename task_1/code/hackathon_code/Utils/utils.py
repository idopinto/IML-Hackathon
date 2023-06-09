import pandas as pd
from datetime import datetime


def load_data(filename, cancel=True):
    if cancel:
        dates_to_parse = ["booking_datetime", "checkin_date",
                          "checkout_date", "hotel_live_date", "cancellation_datetime"]
    else:
        dates_to_parse = ["booking_datetime", "checkin_date",
                          "checkout_date", "hotel_live_date"]
    df = pd.read_csv(filename, parse_dates=dates_to_parse)
    return df


def calculate_percentage_and_count(df, feature):
    # Group by 'days_cancellation_1' column and apply lambda function
    # to calculate the percentage of 'True' values in 'label' column for each group,
    # as well as count the total number of labels in each group
    result = df.groupby(feature)['label'].agg([('y', lambda x: (x == True).mean()), ('count', 'count')]) \
        .reset_index()

    # Rename the columns
    result.columns = ['x', 'y', 'count']

    return result


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
                parsed_policy[i * 2] = int(days)
                parsed_policy[i * 2 + 1] = int(charge)
            else:  # policy_type == 'N'
                parsed_policy[i * 2] = int(days)
                parsed_policy[i * 2 + 1] = (int(charge) / num_nights) * 100

    return parsed_policy
