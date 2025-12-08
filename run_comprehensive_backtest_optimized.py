
import pandas as pd
import itertools
import rotation_engine
from rotation_engine import run_rotation_sip_strategy, run_rotation_sip_strategy_v2_buy_the_dip, _load_multi_fund_nav
import time
from unittest.mock import patch

ALL_FUNDS = [
    "HDFC Flexi Cap Fund - Growth Option - Direct Plan",
    "HDFC Mid Cap Fund - Growth Option - Direct Plan",
    "HDFC Small Cap Fund - Growth Option - Direct Plan",
    "HDFC Large Cap Fund - Growth Option - Direct Plan",
    "HDFC Multi-Asset Fund - Direct Plan - Growth Option",
    "SBI Large Cap Fund - Direct Plan - Growth",
    "SBI Small Cap Fund - Direct Plan - Growth",
    "SBI Midcap Fund - Direct Plan - Growth",
    "SBI Flexicap Fund - Direct Plan - Growth",
    "SBI Multi Asset Allocation Fund - Direct Plan - Growth",
    "UTI Large Cap Fund - Direct Plan - Growth Option",
    "UTI Mid Cap Fund - Growth Option - Direct",
    "UTI Small Cap Fund - Direct Plan - Growth Option",
    "UTI Flexi Cap Fund - Growth Option - Direct",
    "UTI Multi Asset Allocation Fund - Direct Plan - Growth Option",
    "HDFC Nifty 50 Index Fund - Direct Plan",
    "SBI Nifty Index Fund - Direct Plan - Growth",
    "UTI Nifty 50 Index Fund - Growth Option - Direct",
    "HDFC Retirement Savings Fund - Equity Plan - Growth Option - Direct Plan",
    "SBI Retirement Benefit Fund - Aggressive Plan - Direct Plan - Growth",
    "UTI Retirement Fund - Direct Plan",
    "HDFC Gold ETF Fund of Fund - Direct Plan",
    "SBI Gold Fund - Direct Plan - Growth",
    "UTI Gold ETF Fund of Fund - Direct Plan - Growth Option",
]

# Parameters
SIP_AMOUNT = 10000.0
LAMBDA_VOL = 0.5
LAMBDA_DD = 0.5
RISK_FREE_RATE = 0.06

print("Pre-loading data...")
# Load all funds once
all_nav_data = _load_multi_fund_nav(ALL_FUNDS)
print("Data loaded.")

# Mock function to replace _load_multi_fund_nav
def mock_load_multi_fund_nav(fund_names, start_date=None, end_date=None):
    # Select columns
    df = all_nav_data[fund_names].copy()
    # Filter dates if needed (though we usually use full range)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df

# Patch the function in rotation_engine
rotation_engine._load_multi_fund_nav = mock_load_multi_fund_nav

results = []
output_file = "backtest_results.csv"

print("Starting optimized comprehensive backtest...")
start_time = time.time()

combinations = []
for r in range(2, 5): # Sizes 2, 3, 4
    combinations.extend(list(itertools.combinations(ALL_FUNDS, r)))

total_combos = len(combinations)
print(f"Total combinations to test: {total_combos}")

# Write header
pd.DataFrame(columns=[
    "Funds", "Num_Funds", 
    "S1_XIRR", "S1_MaxDD", "S1_FinalValue", 
    "S2_XIRR", "S2_MaxDD", "S2_FinalValue", "Error"
]).to_csv(output_file, index=False)

batch_results = []

for i, combo in enumerate(combinations):
    combo_list = list(combo)
    
    if i % 100 == 0:
        print(f"Processing {i}/{total_combos}...")

    try:
        # Strategy 1
        res1 = run_rotation_sip_strategy(
            fund_names=combo_list,
            sip_amount=SIP_AMOUNT,
            lambda_vol=LAMBDA_VOL,
            lambda_dd=LAMBDA_DD,
            risk_free_rate=RISK_FREE_RATE,
        )
        
        # Strategy 2
        res2 = run_rotation_sip_strategy_v2_buy_the_dip(
            fund_names=combo_list,
            sip_amount=SIP_AMOUNT,
            risk_free_rate=RISK_FREE_RATE,
            max_weight_per_fund=0.65
        )
        
        batch_results.append({
            "Funds": ", ".join(combo_list),
            "Num_Funds": len(combo_list),
            "S1_XIRR": res1.xirr,
            "S1_MaxDD": res1.max_drawdown,
            "S1_FinalValue": res1.final_value,
            "S2_XIRR": res2.xirr,
            "S2_MaxDD": res2.max_drawdown,
            "S2_FinalValue": res2.final_value,
            "Error": ""
        })
        
    except Exception as e:
        # print(f"Error with {combo_list}: {e}")
        batch_results.append({
            "Funds": ", ".join(combo_list),
            "Num_Funds": len(combo_list),
            "Error": str(e)
        })

    # Write incrementally every 50 rows
    if len(batch_results) >= 50:
        pd.DataFrame(batch_results).to_csv(output_file, mode='a', header=False, index=False)
        batch_results = []

# Write remaining
if batch_results:
    pd.DataFrame(batch_results).to_csv(output_file, mode='a', header=False, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Analysis complete in {duration:.2f} seconds.")
print(f"Results saved to {output_file}")
