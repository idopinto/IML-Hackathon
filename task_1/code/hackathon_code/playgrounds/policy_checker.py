from task_1.code.hackathon_code.Utils.utils import cancellation_cost

# Assume an order cost of $500 for 5 nights

print(cancellation_cost("365D100P_100P", "2023-05-10", "2023-06-10", 1000, 5))  # Expected output: 1000
print(cancellation_cost("365D100P_100P", "2023-04-10", "2023-06-10", 1000, 5))  # Expected output: 1000
print(cancellation_cost("365D1N_1N", "2023-05-10", "2023-06-10", 1000, 5))  # Expected output: 200
print(cancellation_cost("365D1N_1N", "2023-04-10", "2023-06-10", 1000, 5))  # Expected output: 200
print(cancellation_cost("30D50P_1N", "2023-05-11", "2023-06-10", 1000, 5))  # Expected output: 500
print(cancellation_cost("30D50P_1N", "2023-04-10", "2023-06-10", 1000, 5))  # Expected output: 0
print(cancellation_cost("60D1N_30D100P_100P", "2023-05-11", "2023-06-10", 1000, 5))  # Expected output: 1000
print(cancellation_cost("60D1N_30D100P_100P", "2023-04-11", "2023-06-10", 1000, 5))  # Expected output: 200
print(cancellation_cost("100P", "2023-06-10", "2023-06-10", 1000, 5))  # Expected output: 1000
print(cancellation_cost("1N", "2023-06-10", "2023-06-10", 1000, 5))  # Expected output: 200

