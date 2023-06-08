from task_1.code.hackathon_code.Utils.utils import parse_policy

assert parse_policy("60D1N_30D100P_100P", 5) == [60, 20, 30, 100, 100]
assert parse_policy("60D1N_30D100P", 5) == [60, 20, 30, 100, -1]
assert parse_policy("60D100P_30D1N", 5) == [60, 100, 30, 20, -1]
assert parse_policy("60D100P_1N", 5) == [60, 100, -1, -1, 20]
assert parse_policy("1N", 5) == [-1, -1, -1, -1, 20]
assert parse_policy("60D1N", 5) == [60, 20, -1, -1, -1]
assert parse_policy("", 5) == [-1, -1, -1, -1, -1]

assert parse_policy("90D2N_15D50P_100P", 10) == [90, 20, 15, 50, 100]
assert parse_policy("60D50P_2N", 4) == [60, 50, -1, -1, 50]
assert parse_policy("2N", 4) == [-1, -1, -1, -1, 50]
assert parse_policy("90D2N", 10) == [90, 20, -1, -1, -1]
assert parse_policy("30D100P", 5) == [30, 100, -1, -1, -1]
assert parse_policy("60D50P_15D2N_1N", 5) == [60, 50, 15, 40, 20]
assert parse_policy("90D100P_30D2N", 10) == [90, 100, 30, 20, -1]
assert parse_policy("45D100P_3N", 3) == [45, 100, -1, -1, 100]
assert parse_policy("3N", 3) == [-1, -1, -1, -1, 100]
assert parse_policy("90D3N", 10) == [90, 30, -1, -1, -1]
assert parse_policy("30D100P_1N", 5) == [30, 100, -1, -1, 20]
assert parse_policy("60D2N_15D50P_1N", 5) == [60, 40, 15, 50, 20]
assert parse_policy("90D100P_30D3N", 10) == [90, 100, 30, 30, -1]



