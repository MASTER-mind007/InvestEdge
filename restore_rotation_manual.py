
import os
import pandas as pd
import numpy as np

file_path = r"c:\Users\navin\Desktop\InvestEdge\rotation_engine.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Check if file is truncated at the expected point
truncated_marker = 'f"Strategy 2: Monthly SIP {sip_amount:.0f}, 50% equal-weight core, trend + buy-the-dip tilt."'
if truncated_marker in content and "return result" not in content.split(truncated_marker)[-1]:
    print("File appears truncated. Restoring...")
    
    # Remove the truncated line to replace it cleanly
    content = content.rsplit(truncated_marker, 1)[0]
    
    # Append the rest of the file
    rest_of_file = '''        f"Strategy 2: Monthly SIP {sip_amount:.0f}, 50% equal-weight core, trend + buy-the-dip tilt."
        ),
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=absolute_return,
        xirr=xirr,
        num_installments=num_installments,
        start_date=portfolio_df['date'].min(),
        end_date=portfolio_df['date'].max(),
        volatility_annual=metrics["volatility_annual"],
        max_drawdown=metrics["max_drawdown"],
        max_dd_peak_date=metrics["max_dd_peak_date"],
        max_dd_trough_date=metrics["max_dd_trough_date"],
        sortino_ratio=metrics["sortino_ratio"],
        alpha_annual=alpha_vs_target,
        beta=beta_vs_target,
        portfolio_nav=portfolio_df,
        allocation_history=allocation_history,
    )

    return result


def run_equal_weight_sip_strategy(
    fund_names: List[str],
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    risk_free_rate: float = 0.06,
) -> StrategyResult:
    """
    Normal SIP: Split SIP amount equally among all funds each month.
    """
    if len(fund_names) < 1:
        raise ValueError("Equal weight strategy needs at least 1 fund.")

    # Reuse v2 logic but with 0 tilt
    # Or simpler: just call v2 with max_weight_per_fund = 1.0/N? 
    # No, v2 has 50% core. 
    # Let's implement simple equal weight.
    
    combined_nav = _load_multi_fund_nav(fund_names, start_date=start_date, end_date=end_date)
    if combined_nav.empty:
        raise ValueError("No NAV data.")
    
    combined_nav = combined_nav.sort_index()
    all_dates = combined_nav.index
    
    if start_date is None: start_date = all_dates.min()
    else: start_date = pd.to_datetime(start_date)
    
    if end_date is None: end_date = all_dates.max()
    else: end_date = pd.to_datetime(end_date)
    
    try:
        sip_dates = pd.date_range(start=start_date, end=end_date, freq="ME").to_pydatetime().tolist()
    except ValueError:
        sip_dates = pd.date_range(start=start_date, end=end_date, freq="M").to_pydatetime().tolist()
    
    units_held = {name: pd.Series(0.0, index=combined_nav.index) for name in fund_names}
    cashflows = []
    cashflow_dates = []
    num_installments = 0
    
    N = len(fund_names)
    per_fund = sip_amount / N
    
    for sip_dt in sip_dates:
        sip_dt = pd.to_datetime(sip_dt)
        future_dates = combined_nav.index[combined_nav.index >= sip_dt]
        if len(future_dates) == 0: continue
        nav_date = future_dates[0]
        
        for name in fund_names:
            nav_price = float(combined_nav.loc[nav_date, name])
            if nav_price > 0:
                units = per_fund / nav_price
                units_held[name].loc[units_held[name].index >= nav_date] += units
        
        cashflows.append(-sip_amount)
        cashflow_dates.append(nav_date)
        num_installments += 1
        
    units_df = pd.DataFrame({name: s for name, s in units_held.items()}, index=combined_nav.index)
    portfolio_values = (units_df * combined_nav).sum(axis=1)
    portfolio_df = pd.DataFrame({"date": portfolio_values.index, "nav": portfolio_values.values})
    
    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])
    
    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0 if total_invested > 0 else np.nan
    xirr = _compute_xirr(cashflows, cashflow_dates)
    
    metrics = _compute_common_metrics(portfolio_df, risk_free_rate, None)
    
    return StrategyResult(
        name="NORMAL SIP (Equal Weight)",
        description=f"Monthly SIP {sip_amount:.0f} split equally among {N} funds.",
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=num_installments,
        start_date=portfolio_df['date'].min(),
        end_date=portfolio_df['date'].max(),
        volatility_annual=metrics["volatility_annual"],
        max_drawdown=metrics["max_drawdown"],
        max_dd_peak_date=metrics["max_dd_peak_date"],
        max_dd_trough_date=metrics["max_dd_trough_date"],
        sortino_ratio=metrics["sortino_ratio"],
        alpha_annual=metrics["alpha_annual"],
        beta=metrics["beta"],
        portfolio_nav=portfolio_df,
        allocation_history=pd.DataFrame(),
    )

def run_baseline_sip_strategy(
    fund_name: str,
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    risk_free_rate: float = 0.06,
) -> StrategyResult:
    """
    Baseline SIP in a single fund.
    """
    return run_equal_weight_sip_strategy(
        fund_names=[fund_name],
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate
    )

def compare_rotation_vs_baseline(
    rotation_funds: List[str],
    baseline_fund: str,
    sip_amount: float,
    lambda_vol: float,
    lambda_dd: float,
    benchmark_name: str,
    risk_free_rate: float,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
):
    # 1. Baseline
    baseline_res = run_baseline_sip_strategy(
        fund_name=baseline_fund,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
    )
    baseline_res.name = f"Baseline SIP in {baseline_fund}"
    
    # 2. Normal Equal Weight
    normal_res = run_equal_weight_sip_strategy(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
    )
    
    # 3. Rotation v1 (Strategy 1)
    rotation_res = run_rotation_sip_strategy(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        lambda_vol=lambda_vol,
        lambda_dd=lambda_dd,
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
    )
    
    # 4. Rotation v2 (Strategy 2)
    rotation_v2_res = run_rotation_sip_strategy_v2_buy_the_dip(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        frequency="M",
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        max_weight_per_fund=0.65,
    )
    
    # Combine results
    results = [baseline_res, rotation_res, normal_res, rotation_v2_res]
    rows = []
    for r in results:
        rows.append({
            "strategy_name": r.name,
            "total_invested": r.total_invested,
            "final_value": r.final_value,
            "absolute_return": r.absolute_return,
            "xirr": r.xirr,
            "num_installments": r.num_installments,
            "volatility_annual": r.volatility_annual,
            "max_drawdown": r.max_drawdown,
            "sortino_ratio": r.sortino_ratio,
        })
    
    comparison_df = pd.DataFrame(rows)
    
    return comparison_df, rotation_res, baseline_res, normal_res, rotation_v2_res

def plot_equity_curves(baseline_res, rotation_res, normal_res, rotation_v2_res, title="Equity Curves"):
    plt.figure(figsize=(10, 6))
    
    for res, style in [
        (baseline_res, "-"),
        (rotation_res, "-"),
        (normal_res, "--"),
        (rotation_v2_res, "-.")
    ]:
        df = res.portfolio_nav
        if df.empty: continue
        plt.plot(df["date"], df["nav"], label=res.name, linestyle=style)
        
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_drawdown_curves(baseline_res, rotation_res, normal_res, rotation_v2_res, title="Drawdown Curves"):
    plt.figure(figsize=(10, 6))
    
    for res, style in [
        (baseline_res, "-"),
        (rotation_res, "-"),
        (normal_res, "--"),
        (rotation_v2_res, "-.")
    ]:
        df = res.portfolio_nav
        if df.empty: continue
        
        # Calculate drawdown
        peak = df["nav"].cummax()
        dd = df["nav"] / peak - 1.0
        
        plt.plot(df["date"], dd, label=res.name, linestyle=style)
        
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_allocation_history(strategy_res, as_weights=True, kind="area", title="Allocation History"):
    df = strategy_res.allocation_history
    if df is None or df.empty:
        return
        
    # Pivot
    pivot_df = df.pivot(index="month", columns="fund_name", values="weight" if as_weights else "amount")
    pivot_df = pivot_df.fillna(0.0)
    
    pivot_df.plot(kind=kind, figsize=(10, 6), stacked=True, alpha=0.7)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Weight" if as_weights else "Amount")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

'''
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content + rest_of_file)
    print("Restored rotation_engine.py")

else:
    print("File does not appear truncated at the expected point.")

