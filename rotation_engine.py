# rotation_engine.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from returns_engine import (
    load_clean_nav,
    compute_volatility,
    compute_max_drawdown,
    compute_sortino_ratio,
    compute_alpha_beta,
)
from signals import compute_signals_for_funds


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_RATE = 0.12  # 12% annual target / hurdle for 'alpha'

# Result container

@dataclass
class StrategyResult:
  
    name: str
    description: str
    total_invested: float
    final_value: float
    absolute_return: float
    xirr: float
    num_installments: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    volatility_annual: Optional[float]
    max_drawdown: Optional[float]
    max_dd_peak_date: Optional[pd.Timestamp]
    max_dd_trough_date: Optional[pd.Timestamp]
    sortino_ratio: Optional[float]
    alpha_annual: Optional[float]
    beta: Optional[float]
    portfolio_nav: pd.DataFrame
    allocation_history: pd.DataFrame


# Internal helpers


def _ensure_portfolio_daily_return(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    
    df = portfolio_df.copy().sort_values("date")
    if "daily_return" not in df.columns:
        df["nav_prev"] = df["nav"].shift(1)
        df["daily_return"] = df["nav"] / df["nav_prev"] - 1
        df.loc[df["nav_prev"].isna(), "daily_return"] = np.nan
        df = df.drop(columns=["nav_prev"])
    return df


def _compute_xirr(cashflows: List[float], dates: List[pd.Timestamp]) -> float:
   
    if len(cashflows) != len(dates):
        raise ValueError("cashflows and dates must have same length")

    if len(cashflows) < 2:
        return float("nan")

    t0 = dates[0]
    year_fracs = np.array([(d - t0).days / 365.25 for d in dates], dtype=float)
    cf = np.array(cashflows, dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(cf / (1 + rate) ** year_fracs))

    low, high = -0.99, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


def _load_multi_fund_nav(
    fund_names: List[str],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
  
    nav_data = {}
    for name in fund_names:
        nav_obj = load_clean_nav(name)
        df = nav_obj.df[["date", "nav"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Rename nav col to fund name
        df = df.rename(columns={"nav": name})
        nav_data[name] = df

    # Combine
    if not nav_data:
        return pd.DataFrame()
    
    combined = pd.concat(nav_data.values(), axis=1)
    combined = combined.sort_index()
    
    if start_date:
        combined = combined[combined.index >= pd.to_datetime(start_date)]
    if end_date:
        combined = combined[combined.index <= pd.to_datetime(end_date)]
        
    # Forward fill missing data (e.g. holidays differ)
    combined = combined.ffill()
    
    return combined


def _compute_trend_and_dip_scores(
    combined_nav: pd.DataFrame,
    nav_date: pd.Timestamp,
    min_history_days: int = 250,
) -> Optional[Dict[str, Dict[str, float]]]:
   
    idx = combined_nav.index
    if nav_date not in idx:
        # Align to the nearest available date on or after nav_date
        future_dates = idx[idx >= nav_date]
        if len(future_dates) == 0:
            return None
        nav_date = future_dates[0]

    # 12M and 6M windows (approx 252 and 126 trading days)
    win12_start = nav_date - pd.Timedelta(days=365)
    win6_start = nav_date - pd.Timedelta(days=182)

    window12 = combined_nav.loc[(idx > win12_start) & (idx <= nav_date)]
    window6 = combined_nav.loc[(idx > win6_start) & (idx <= nav_date)]

    # Basic sanity: need some history
    if len(window12) < min_history_days * 0.4 or len(window6) < min_history_days * 0.2:
        return None

    scores: Dict[str, Dict[str, float]] = {}

    # --- Returns ---
    start12 = window12.iloc[0]
    end12 = window12.iloc[-1]
    R12 = end12 / start12 - 1.0

    start6 = window6.iloc[0]
    end6 = window6.iloc[-1]
    R6 = end6 / start6 - 1.0

    # --- 6M drawdown per fund ---
    DD6: Dict[str, float] = {}
    for col in window6.columns:
        s = window6[col].dropna()
        if s.empty:
            DD6[col] = 0.0
            continue
        peak = s.cummax()
        dd = s / peak - 1.0
        DD6[col] = float(dd.min())  # <= 0

    for col in combined_nav.columns:
        r12 = float(R12.get(col, np.nan))
        r6 = float(R6.get(col, np.nan))
        dd6 = float(DD6.get(col, 0.0))

        if np.isnan(r12) or np.isnan(r6):
            continue

        # Strength: 0.6 * 12M + 0.4 * 6M
        strength = 0.6 * r12 + 0.4 * r6

        # Dip: only if 12M trend positive, else 0
        if r12 > 0:
            dip_raw = max(0.0, -dd6)  # positive number when in drawdown
        else:
            dip_raw = 0.0

        # Cap dip at 30% (0.30)
        dip_capped = min(dip_raw, 0.30)

        scores[col] = {
            "R12": r12,
            "R6": r6,
            "strength": strength,
            "dip_raw": dip_raw,
            "dip_capped": dip_capped,
        }

    if not scores:
        return None

    return scores


def _compute_common_metrics(
    portfolio_df: pd.DataFrame,
    risk_free_rate: float,
    benchmark_name: Optional[str] = None,
) -> Dict[str, Any]:
   
    # Ensure daily returns
    df = _ensure_portfolio_daily_return(portfolio_df)
    
    # Filter out initial zero/NaN periods for risk metrics to avoid inf/nan
    metrics_df = df[df["nav"] > 1e-9].copy()
    metrics_df = _ensure_portfolio_daily_return(metrics_df)

    vol = compute_volatility(metrics_df)
    mdd, peak, trough = compute_max_drawdown(metrics_df)
    sortino = compute_sortino_ratio(metrics_df, risk_free_rate=risk_free_rate)

    alpha_annual = None
    beta = None

    if benchmark_name is not None:
        try:
            bench_obj = load_clean_nav(benchmark_name)
            bench_df = bench_obj.df[["date", "nav"]].copy()
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            
            # Align dates
            start_dt = metrics_df["date"].min()
            end_dt = metrics_df["date"].max()
            bench_df = bench_df[(bench_df["date"] >= start_dt) & (bench_df["date"] <= end_dt)].copy()
            
            if not bench_df.empty:
                ab = compute_alpha_beta(metrics_df, bench_df, risk_free_rate=risk_free_rate)
                alpha_annual = ab.get("alpha_annual")
                beta = ab.get("beta")
        except Exception:
            # If benchmark load fails or other issue, just skip alpha/beta
            pass

    return {
        "volatility_annual": vol,
        "max_drawdown": mdd,
        "max_dd_peak_date": peak,
        "max_dd_trough_date": trough,
        "sortino_ratio": sortino,
        "alpha_annual": alpha_annual,
        "beta": beta,
        "metrics_df": metrics_df,
    }


# Core rotation backtest


def run_rotation_sip_strategy(
    fund_names: List[str],
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    lambda_vol: float = 0.5,
    lambda_dd: float = 0.5,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
    """
    Backtest the rotation strategy among a small set of funds.

    RULE:
    - Each month, invest a fixed SIP amount (sip_amount).
    - Each selected fund must receive at least 10% of the SIP.
    - The remaining SIP is allocated to funds in proportion to
      a composite score based on multi-horizon returns and risk.

    Composite score for fund i:
        Score_i = 0.2 * R1 + 0.3 * R3 + 0.5 * R6
                  - lambda_vol * VOL6
                  - lambda_dd * |DD6|

    where:
        R1, R3, R6 = 1 / 3 / 6-month returns
        VOL6       = 6-month annualised volatility (daily data)
        DD6        = 6-month max drawdown (negative)

    SIP logic:
        base_amt = 0.10 * sip_amount  for each fund
        extra_pool = sip_amount - base_amt * N

        s_i = max(Score_i, 0)
        if sum(s_i) > 0:
            w_extra_i = s_i / sum(s_i)
        else:
            w_extra_i = 1 / N (equal split)

        total_amt_i = base_amt + extra_pool * w_extra_i

    Parameters
    ----------
    fund_names : list of str
        User-selected funds (ideally 2–4 funds, one per category).
    sip_amount : float
        Monthly SIP amount in rupees.
    start_date, end_date : optional
        Optional overall backtest window; if None, use full common overlap.
    lambda_vol, lambda_dd : float
        Penalties for volatility and drawdown in the score.
    benchmark_name : str, optional
        If provided, used for alpha/beta computation.
    risk_free_rate : float
        Annual risk-free rate (e.g., 0.06 for 6%).

   
    """
    if len(fund_names) < 2:
        raise ValueError("Rotation strategy requires at least 2 funds.")
    if len(fund_names) > 4:
        raise ValueError("Rotation strategy currently supports up to 4 funds.")

    N = len(fund_names)

    
    # 1) Load daily NAV for each fund
    
    nav_data: Dict[str, pd.DataFrame] = {}
    for name in fund_names:
        nav_obj = load_clean_nav(name)
        df_nav = nav_obj.df[["date", "nav"]].copy()
        df_nav["date"] = pd.to_datetime(df_nav["date"])
        df_nav = df_nav.sort_values("date").reset_index(drop=True)
        nav_data[name] = df_nav

    # Determine global date range if not provided
    all_start_dates = [df["date"].min() for df in nav_data.values()]
    all_end_dates = [df["date"].max() for df in nav_data.values()]

    if start_date is None:
        start_date = max(all_start_dates)
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = min(all_end_dates)
    else:
        end_date = pd.to_datetime(end_date)

    if start_date >= end_date:
        raise ValueError("Invalid backtest window: no common overlap between funds.")

    # Trim daily NAVs to common window
    for name in fund_names:
        df_nav = nav_data[name]
        df_nav = df_nav[(df_nav["date"] >= start_date) & (df_nav["date"] <= end_date)].copy()
        df_nav = df_nav.sort_values("date").reset_index(drop=True)
        nav_data[name] = df_nav

        # 2) Compute monthly signals for all selected funds
    
    signals_df = compute_signals_for_funds(
        fund_names=fund_names,
        start_date=start_date,
        end_date=end_date,
        min_window_days=30,
    )
    # 'signals_df' columns: ['fund_name','date','nav','R1','R3','R6','VOL6','DD6']

    sip_months = sorted(signals_df["date"].unique().tolist())
    if not sip_months:
        raise ValueError("No monthly signals available within the selected window.")

    
    # 3) Prepare daily index and structures for units per fund
    
    # Union of all daily dates across funds
    all_dates = sorted(
        set(
            d
            for df in nav_data.values()
            for d in df["date"].tolist()
            if start_date <= d <= end_date
        )
    )
    if not all_dates:
        raise ValueError("No daily NAV data in the chosen window for selected funds.")

    daily_index = pd.to_datetime(all_dates)
    daily_index = pd.Index(daily_index)

    # For each fund, create series for daily nav (forward-filled) and units_delta
    nav_aligned: Dict[str, pd.Series] = {}
    units_delta: Dict[str, pd.Series] = {}

    for name in fund_names:
        df_nav = nav_data[name].set_index("date").reindex(daily_index)
        # Forward-fill NAVs (assumes missing days are non-trading days)
        df_nav["nav"] = df_nav["nav"].ffill()
        nav_aligned[name] = df_nav["nav"]
        # Units delta starts at 0 for each day
        units_delta[name] = pd.Series(0.0, index=daily_index)

    
    # 4) Monthly SIP loop – apply 10% base + score-based extra
    
    allocation_records = []
    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []

    for m_date in sip_months:
        m_date = pd.to_datetime(m_date)

        # Base amount and extra pool
        base_amt = 0.10 * sip_amount
        base_total = base_amt * N
        extra_pool = sip_amount - base_total
        if extra_pool < -1e-6:
            raise ValueError(
                "Extra pool is negative. Check SIP amount and 10% per fund rule."
            )

        # Gather scores per fund
        scores: Dict[str, float] = {}
        for name in fund_names:
            row = signals_df[
                (signals_df["fund_name"] == name) & (signals_df["date"] == m_date)
            ]
            if row.empty:
                # If no signals for this month (very unlikely after filtering), treat as 0
                scores[name] = 0.0
                continue

            r1 = float(row["R1"].iloc[0]) if not pd.isna(row["R1"].iloc[0]) else 0.0
            r3 = float(row["R3"].iloc[0]) if not pd.isna(row["R3"].iloc[0]) else 0.0
            r6 = float(row["R6"].iloc[0]) if not pd.isna(row["R6"].iloc[0]) else 0.0
            vol6 = float(row["VOL6"].iloc[0]) if not pd.isna(row["VOL6"].iloc[0]) else 0.0
            dd6 = float(row["DD6"].iloc[0]) if not pd.isna(row["DD6"].iloc[0]) else 0.0

            score = 0.2 * r1 + 0.3 * r3 + 0.5 * r6 - lambda_vol * vol6 - lambda_dd * abs(dd6)
            scores[name] = score

        # Convert scores to non-negative values for extra allocation
        s_pos = {name: max(score, 0.0) for name, score in scores.items()}
        s_sum = sum(s_pos.values())

        if s_sum > 0:
            extra_weights = {name: v / s_sum for name, v in s_pos.items()}
        else:
            # If all scores <= 0, split extra equally
            extra_weights = {name: 1.0 / N for name in fund_names}

        # Compute allocation per fund for this month
        alloc_for_month: Dict[str, float] = {}
        for name in fund_names:
            extra_amt = extra_pool * extra_weights[name]
            total_amt = base_amt + extra_amt
            alloc_for_month[name] = total_amt

        # Record one cash outflow (full SIP) at monthly signal date
        cashflows.append(-sip_amount)
        cashflow_dates.append(m_date)

        # Convert rupees -> units per fund, and update units_delta on daily index
        for name in fund_names:
            # Use the last available NAV on or before m_date
            df_nav = nav_data[name]
            nav_row = df_nav[df_nav["date"] <= m_date]
            if nav_row.empty:
                continue  # should not happen after trimming
            trade_date = nav_row["date"].iloc[-1]
            nav_price = float(nav_row["nav"].iloc[-1])

            units = alloc_for_month[name] / nav_price

            # Update units from trade_date onwards
            units_delta[name].loc[units_delta[name].index >= trade_date] += units

            allocation_records.append(
                {
                    "month": m_date,
                    "fund_name": name,
                    "trade_date": trade_date,
                    "amount": alloc_for_month[name],
                    "units": units,
                    "score": scores[name],
                }
            )

    num_installments = len(cashflows)


    # 5) Build daily portfolio NAV
    
    portfolio_values = pd.Series(0.0, index=daily_index)

    for name in fund_names:
        units_series = units_delta[name]
        # units_series is already cumulative because we added to all future dates
        portfolio_values += units_series * nav_aligned[name]

    portfolio_df = pd.DataFrame(
        {"date": portfolio_values.index, "nav": portfolio_values.values}
    )

    # Final value & XIRR
    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0 if total_invested > 0 else float("nan")
    xirr = _compute_xirr(cashflows, cashflow_dates)

    
    # 6) Risk metrics
    
    metrics = _compute_common_metrics(
        portfolio_df=portfolio_df,
        risk_free_rate=risk_free_rate,
        benchmark_name=benchmark_name,
    )
    
    
    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    allocation_history = pd.DataFrame(allocation_records)
    if not allocation_history.empty:
        allocation_history = allocation_history.sort_values(["month", "fund_name"]).reset_index(drop=True)

    result = StrategyResult(
        name="Rotation SIP Strategy (10% floor per fund)",
        description=(
            f"Monthly SIP of {sip_amount:.0f} with 10% base allocation to each fund and "
            f"score-based tilt among: {', '.join(fund_names)}. "
            f"Score = 0.2*R1 + 0.3*R3 + 0.5*R6 - {lambda_vol}*VOL6 - {lambda_dd}*|DD6|."
        ),
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=num_installments,
        start_date=portfolio_df["date"].min(),
        end_date=portfolio_df["date"].max(),
        volatility_annual=metrics["volatility_annual"],
        max_drawdown=metrics["max_drawdown"],
        max_dd_peak_date=metrics["max_dd_peak_date"],
        max_dd_trough_date=metrics["max_dd_trough_date"],
        sortino_ratio=metrics["sortino_ratio"],
        alpha_annual=metrics["alpha_annual"],
        beta=metrics["beta"],
        portfolio_nav=portfolio_df,
        allocation_history=allocation_history,
    )

    return result


def run_rotation_sip_strategy_v2_buy_the_dip(
    fund_names: List[str],
    sip_amount: float = 100000.0,
    frequency: str = "M",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    risk_free_rate: float = 0.06,
    max_weight_per_fund: float = 0.65,
) -> StrategyResult:
    """
    Rotation v2: 50% equal-weight + 25% trend tilt + 25% buy-the-dip tilt.

    Monthly rules:
      - 50% of SIP is split equally among all funds every month.
      - 25% is allocated by a 'trend' score (12M + 6M returns, relative).
      - 25% is allocated by a 'dip' score (drawdown over 6M for funds with positive 12M trend).
      - Optional cap on per-fund weight (max_weight_per_fund).

    This strategy does NOT assume we can perfectly predict winners, so:
      - Equal-weight core stays large.
      - Tilts are bounded and rules-based.
    """

    if len(fund_names) < 2:
        raise ValueError("Rotation v2 needs at least 2 funds.")
    if sip_amount <= 0:
        raise ValueError("sip_amount must be positive.")

    # Load combined NAV matrix for all funds
    combined_nav = _load_multi_fund_nav(fund_names, start_date=start_date, end_date=end_date)
    if combined_nav.empty:
        raise ValueError("No NAV data available for the given funds / period.")

    combined_nav = combined_nav.sort_index()
    all_dates = combined_nav.index

    # If start/end not provided, infer from combined NAV
    if start_date is None:
        start_date = all_dates.min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = all_dates.max()
    else:
        end_date = pd.to_datetime(end_date)

    # SIP schedule: monthly
    sip_dates = pd.date_range(start=start_date, end=end_date, freq=frequency).to_pydatetime().tolist()
    if not sip_dates:
        raise ValueError("No SIP dates generated for the given period.")

    # Track units held over time: units_held[name][date]
    units_held = {name: pd.Series(0.0, index=combined_nav.index) for name in fund_names}

    # For XIRR
    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []
    num_installments = 0

    # For allocation history
    alloc_rows: List[Dict] = []

    N = len(fund_names)

    for sip_dt in sip_dates:
        sip_dt = pd.to_datetime(sip_dt)
        # Align to nearest available NAV date on or after sip_dt
        future_dates = combined_nav.index[combined_nav.index >= sip_dt]
        if len(future_dates) == 0:
            continue
        nav_date = future_dates[0]

        # ---- Base equal-weight: 50% of SIP ----
        base_amt_total = 0.50 * sip_amount
        base_per_fund = base_amt_total / N

        # ---- Trend & Dip scores ----
        scores = _compute_trend_and_dip_scores(combined_nav, nav_date)
        # Default: equal weights for both tilts
        trend_weights = {name: 1.0 / N for name in fund_names}
        dip_weights = {name: 1.0 / N for name in fund_names}

        if scores is not None:
            # Build arrays for strength and dip
            # Identify funds with valid scores
            valid_funds = [n for n in fund_names if n in scores]
            
            if not valid_funds:
                 # Fallback if no funds have valid scores
                 trend_weights = {name: 1.0 / len(fund_names) for name in fund_names}
                 dip_weights = {name: 1.0 / len(fund_names) for name in fund_names}
            else:
                strength_vals = np.array([scores[n]["strength"] for n in valid_funds])
                dip_vals = np.array([scores[n]["dip_capped"] for n in valid_funds])
                
                # --- Trend tilt: based on relative strength ---
                s_min = float(np.min(strength_vals))
                s_max = float(np.max(strength_vals))
                if s_max - s_min > 1e-8:
                    strength_norm = (strength_vals - s_min) / (s_max - s_min)
                    exp_s = np.exp(strength_norm)
                    trend_w = exp_s / exp_s.sum()
                else:
                    trend_w = np.ones_like(strength_vals) / len(valid_funds)
                    
                # --- Dip tilt: buy dips only if positive trend ---
                if dip_vals.sum() > 1e-8:
                    dip_w = dip_vals / dip_vals.sum()
                else:
                    dip_w = np.ones_like(dip_vals) / len(valid_funds)
                    
                # Map back to full fund list
                trend_weights = {name: 0.0 for name in fund_names}
                dip_weights = {name: 0.0 for name in fund_names}
                
                for name, w in zip(valid_funds, trend_w):
                    trend_weights[name] = float(w)
                for name, w in zip(valid_funds, dip_w):
                    dip_weights[name] = float(w)
        trend_amt_total = 0.25 * sip_amount
        dip_amt_total = 0.25 * sip_amount

        # Start with base equal-weight part
        alloc_amt: Dict[str, float] = {name: base_per_fund for name in fund_names}

        # Add trend tilt
        for name in fund_names:
            alloc_amt[name] += trend_amt_total * trend_weights[name]

        # Add dip tilt
        for name in fund_names:
            alloc_amt[name] += dip_amt_total * dip_weights[name]

        # Optional cap per fund and re-normalise
        if max_weight_per_fund is not None:
            # Convert to weights, cap, then renormalise
            weights_raw = np.array([alloc_amt[name] / sip_amount for name in fund_names])
            weights_capped = np.minimum(weights_raw, max_weight_per_fund)
            total_w = weights_capped.sum()
            if total_w <= 0:
                # Fallback: equal if everything got zeroed (should not happen usually)
                weights_final = np.ones_like(weights_capped) / N
            else:
                weights_final = weights_capped / total_w
            for name, w in zip(fund_names, weights_final):
                alloc_amt[name] = float(w * sip_amount)

        # Sanity: ensure final allocations sum (approximately) to SIP amount
        total_alloc = sum(alloc_amt.values())
        if abs(total_alloc - sip_amount) > 1e-3:
            # Small renormalisation to avoid drift
            factor = sip_amount / total_alloc
            for name in fund_names:
                alloc_amt[name] *= factor

        # ---- Execute SIP: convert rupees to units, update units_held ----
        for name in fund_names:
            nav_price = float(combined_nav.loc[nav_date, name])
            amt = alloc_amt[name]
            if nav_price <= 0 or amt <= 0:
                units = 0.0
            else:
                units = amt / nav_price

            # Add units from nav_date forward
            units_held[name].loc[units_held[name].index >= nav_date] += units

            alloc_rows.append(
                {
                    "month": nav_date.normalize(),
                    "trade_date": nav_date,
                    "fund_name": name,
                    "amount": amt,
                    "units": units,
                    "weight": amt / sip_amount,
                    "trend_weight": trend_weights.get(name, 1.0 / N),
                    "dip_weight": dip_weights.get(name, 1.0 / N),
                    "R12": scores[name]["R12"] if scores is not None and name in scores else np.nan,
                    "R6": scores[name]["R6"] if scores is not None and name in scores else np.nan,
                    "strength": scores[name]["strength"] if scores is not None and name in scores else np.nan,
                    "dip_capped": scores[name]["dip_capped"] if scores is not None and name in scores else np.nan,
                }
            )

        cashflows.append(-sip_amount)
        cashflow_dates.append(nav_date)
        num_installments += 1

    if num_installments == 0:
        raise ValueError("No SIP installments executed in rotation v2 (check dates / data).")

    # ---- Build portfolio NAV over time ----
    units_df = pd.DataFrame({name: s for name, s in units_held.items()}, index=combined_nav.index)
    portfolio_values = (units_df * combined_nav).sum(axis=1)

    portfolio_df = pd.DataFrame({"date": portfolio_values.index, "nav": portfolio_values.values})

    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    absolute_return = final_value / total_invested - 1.0 if total_invested > 0 else np.nan
    xirr = _compute_xirr(cashflows, cashflow_dates)

    # ---- Risk metrics ----
    # For v2, we calculate standard metrics, but Alpha is vs Target Rate
    metrics = _compute_common_metrics(
        portfolio_df=portfolio_df,
        risk_free_rate=risk_free_rate,
        benchmark_name=None, # We calculate alpha manually below
    )
    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    # 'Alpha' vs 12% target (NOT CAPM alpha)
    alpha_vs_target = xirr - TARGET_RATE
    beta_vs_target = None

    allocation_history = pd.DataFrame(alloc_rows)

    return StrategyResult(
        name="Rotation SIP v2 – 50% Equal + Trend + Buy-the-Dip",
        description=(
            f"Monthly SIP {sip_amount:.0f} with 50% equal-weight core, "
            f"25% trend tilt (12M+6M) and 25% buy-the-dip tilt, "
            f"max {max_weight_per_fund:.0%} per fund."
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



# Baseline SIP strategy (single fund)


def run_baseline_sip_strategy(
    fund_name: str,
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
   
    nav_obj = load_clean_nav(fund_name)
    df_nav = nav_obj.df[["date", "nav"]].copy()
    df_nav["date"] = pd.to_datetime(df_nav["date"])
    df_nav = df_nav.sort_values("date").reset_index(drop=True)

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_nav = df_nav[df_nav["date"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_nav = df_nav[df_nav["date"] <= end_date]

    if df_nav.empty:
        raise ValueError(f"No NAV data for baseline SIP in '{fund_name}' in chosen window.")

    # Month-end dates and NAVs
    monthly_nav = (
        df_nav.set_index("date")["nav"]
        .resample("M")
        .last()
        .dropna()
    )
    monthly_df = monthly_nav.to_frame(name="nav").reset_index()  # ['date','nav']

    # Daily index for full period
    all_dates = df_nav["date"].tolist()
    daily_index = pd.Index(sorted(pd.to_datetime(all_dates)))

    # Align NAV daily (ffill)
    df_nav_daily = df_nav.set_index("date").reindex(daily_index)
    df_nav_daily["nav"] = df_nav_daily["nav"].ffill()
    nav_series = df_nav_daily["nav"]

    # Prepare units delta over time
    units_delta = pd.Series(0.0, index=daily_index)

    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []

    # SIP each month at that month's date and nav
    for _, row in monthly_df.iterrows():
        m_date = pd.to_datetime(row["date"])
        nav_price = float(row["nav"])
        units = sip_amount / nav_price

        # Add units from m_date onwards
        units_delta.loc[units_delta.index >= m_date] += units

        cashflows.append(-sip_amount)
        cashflow_dates.append(m_date)

    if not cashflows:
        raise ValueError("No SIP installments executed in baseline strategy.")

    # Build portfolio NAV
    portfolio_values = units_delta * nav_series
    portfolio_df = pd.DataFrame({"date": portfolio_values.index, "nav": portfolio_values.values})

    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0 if total_invested > 0 else float("nan")
    xirr = _compute_xirr(cashflows, cashflow_dates)

    metrics = _compute_common_metrics(
        portfolio_df=portfolio_df,
        risk_free_rate=risk_free_rate,
        benchmark_name=benchmark_name,
    )
    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    # Baseline has no allocation_history; keep empty frame
    allocation_history = pd.DataFrame(columns=["month", "fund_name", "trade_date", "amount", "units", "score"])

    return StrategyResult(
        name=f"Baseline SIP in {fund_name}",
        description=f"Monthly SIP of {sip_amount:.0f} into {fund_name}, no rotation.",
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=len([cf for cf in cashflows if cf < 0]),
        start_date=portfolio_df["date"].min(),
        end_date=portfolio_df["date"].max(),
        volatility_annual=metrics["volatility_annual"],
        max_drawdown=metrics["max_drawdown"],
        max_dd_peak_date=metrics["max_dd_peak_date"],
        max_dd_trough_date=metrics["max_dd_trough_date"],
        sortino_ratio=metrics["sortino_ratio"],
        alpha_annual=metrics["alpha_annual"],
        beta=metrics["beta"],
        portfolio_nav=portfolio_df,
        allocation_history=allocation_history,
    )



def run_equal_weight_sip_strategy(
    fund_names: List[str],
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
    """
    Normal SIP:
    - Equal distribution of SIP amount among selected funds.
    - No rotation, just fixed equal weight.
    """
    if not fund_names:
        raise ValueError("fund_names list cannot be empty for equal weight SIP.")

    amount_per_fund = sip_amount / len(fund_names)
    
    # Run baseline SIP for each fund
    results = []
    for fund in fund_names:
        res = run_baseline_sip_strategy(
            fund_name=fund,
            sip_amount=amount_per_fund,
            start_date=start_date,
            end_date=end_date,
            benchmark_name=None, # We'll compute alpha/beta on the aggregate
            risk_free_rate=risk_free_rate,
        )
        results.append(res)

       
    # Start with the first fund's portfolio
    combined_df = results[0].portfolio_nav[["date", "nav"]].copy()
    combined_df = combined_df.set_index("date")
    
    for res in results[1:]:
        df = res.portfolio_nav[["date", "nav"]].copy().set_index("date")
        # Align and sum
        combined_df = combined_df.join(df, how="inner", lsuffix="_1", rsuffix="_2")
        combined_df["nav"] = combined_df["nav_1"] + combined_df["nav_2"]
        combined_df = combined_df[["nav"]]

    combined_df = combined_df.reset_index()
    
    # Re-calculate metrics on combined portfolio
    final_value = float(combined_df["nav"].iloc[-1])
    
    # Total invested is simple sum
    total_invested = sum(r.total_invested for r in results)
    abs_return = final_value / total_invested - 1.0 if total_invested > 0 else float("nan")
    
   
    all_cashflows = []
    all_dates = []
    
    num_installments = results[0].num_installments # Assuming all ran for same duration
    
  
    start_dt = combined_df["date"].min()
    end_dt = combined_df["date"].max()
    
    # Generate monthly dates
    sip_dates = pd.date_range(start=start_dt, end=end_dt, freq="M")
    # This might not match exactly if funds have different holidays, but close enough.
    
    cf = [-sip_amount] * len(sip_dates)
    dates = list(sip_dates)
    
    # Add final value
    cf.append(final_value)
    dates.append(end_dt)
    
    xirr = _compute_xirr(cf, dates)
    
    # Risk metrics
    # Risk metrics
    metrics = _compute_common_metrics(
        portfolio_df=combined_df,
        risk_free_rate=risk_free_rate,
        benchmark_name=benchmark_name,
    )
    combined_df = _ensure_portfolio_daily_return(combined_df)

    return StrategyResult(
        name="NORMAL SIP (Equal Weight)",
        description=f"Equal SIP of {sip_amount:.0f} split among {len(fund_names)} funds.",
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=len(sip_dates),
        start_date=start_dt,
        end_date=end_dt,
        volatility_annual=metrics["volatility_annual"],
        max_drawdown=metrics["max_drawdown"],
        max_dd_peak_date=metrics["max_dd_peak_date"],
        max_dd_trough_date=metrics["max_dd_trough_date"],
        sortino_ratio=metrics["sortino_ratio"],
        alpha_annual=metrics["alpha_annual"],
        beta=metrics["beta"],
        portfolio_nav=combined_df,
        allocation_history=pd.DataFrame(), # No dynamic allocation
    )


# Comparison helper: rotation vs baseline vs normal

def compare_rotation_vs_baseline(
    rotation_funds: List[str],
    baseline_fund: str,
    sip_amount: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    lambda_vol: float = 0.5,
    lambda_dd: float = 0.5,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, StrategyResult, StrategyResult, StrategyResult, StrategyResult]:
    """
    Run four strategies:
      1. Baseline SIP (single fund)
      2. Normal SIP (Equal weight among rotation funds)
      3. Rotation SIP v1 (10% floor + score-based tilt)
      4. Rotation SIP v2 (50% equal + trend + dip)
    """
    baseline_result = run_baseline_sip_strategy(
        fund_name=baseline_fund,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
    )

    normal_result = run_equal_weight_sip_strategy(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
    )

    rotation_result = run_rotation_sip_strategy(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        lambda_vol=lambda_vol,
        lambda_dd=lambda_dd,
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
    )

    rotation_v2_result = run_rotation_sip_strategy_v2_buy_the_dip(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
    )

    rows = []
    for r in [baseline_result, normal_result, rotation_result, rotation_v2_result]:
        rows.append({
            "strategy_name": r.name,
            "description": r.description,
            "total_invested": r.total_invested,
            "final_value": r.final_value,
            "absolute_return": r.absolute_return,
            "xirr": r.xirr,
            "volatility_annual": r.volatility_annual,
            "max_drawdown": r.max_drawdown,
            "sortino_ratio": r.sortino_ratio,
            "alpha_annual": r.alpha_annual,
            "beta": r.beta,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "num_installments": r.num_installments,
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df, rotation_result, baseline_result, normal_result, rotation_v2_result

# Plotting helpers

def plot_equity_curves(
    baseline_result: StrategyResult,
    rotation_result: StrategyResult,
    normal_result: Optional[StrategyResult] = None,
    rotation_v2_result: Optional[StrategyResult] = None,
    title: str = "Portfolio Value Over Time",
):
    """
    Plot strategy equity curves on the same chart using matplotlib.
    """
    plt.figure(figsize=(10, 5))
    b = baseline_result.portfolio_nav
    r = rotation_result.portfolio_nav
    
    plt.plot(b["date"], b["nav"], label=baseline_result.name)
    plt.plot(r["date"], r["nav"], label=rotation_result.name)
    
    if normal_result:
        n = normal_result.portfolio_nav
        plt.plot(n["date"], n["nav"], label=normal_result.name, linestyle="--")

    if rotation_v2_result:
        v2 = rotation_v2_result.portfolio_nav
        plt.plot(v2["date"], v2["nav"], label=rotation_v2_result.name, linestyle="-.")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_drawdown_curves(
    baseline_result: StrategyResult,
    rotation_result: StrategyResult,
    normal_result: Optional[StrategyResult] = None,
    rotation_v2_result: Optional[StrategyResult] = None,
    title: str = "Drawdown Comparison",
):
    """
    Plot drawdown curves (in %) for both strategies.
    """
    def compute_drawdown_series(df: pd.DataFrame) -> pd.Series:
        nav = df["nav"]
        peak = nav.cummax()
        dd = nav / peak - 1.0
        return dd

    b = baseline_result.portfolio_nav.copy()
    r = rotation_result.portfolio_nav.copy()

    b_dd = compute_drawdown_series(b)
    r_dd = compute_drawdown_series(r)

    plt.figure(figsize=(10, 5))
    plt.plot(b["date"], b_dd * 100, label=baseline_result.name)
    plt.plot(r["date"], r_dd * 100, label=rotation_result.name)
    
    if normal_result:
        n = normal_result.portfolio_nav.copy()
        n_dd = compute_drawdown_series(n)
        plt.plot(n["date"], n_dd * 100, label=normal_result.name, linestyle="--")

    if rotation_v2_result:
        v2 = rotation_v2_result.portfolio_nav.copy()
        v2_dd = compute_drawdown_series(v2)
        plt.plot(v2["date"], v2_dd * 100, label=rotation_v2_result.name, linestyle="-.")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()



def plot_allocation_history(
    rotation_result: StrategyResult,
    as_weights: bool = True,
    kind: str = "area",
    title: str = "Monthly SIP Allocation – Rotation Strategy",
):
   
    alloc = rotation_result.allocation_history.copy()
    if alloc is None or alloc.empty:
        raise ValueError("allocation_history is empty; did you run the rotation strategy?")

    # Ensure datetime and tidy pivot
    alloc["month"] = pd.to_datetime(alloc["month"])
    monthly = (
        alloc.groupby(["month", "fund_name"])["amount"]
        .sum()
        .unstack("fund_name")
        .fillna(0.0)
        .sort_index()
    )

    if as_weights:
        monthly = monthly.div(monthly.sum(axis=1), axis=0).fillna(0.0)
        y_label = "Allocation weight"
    else:
        y_label = "Monthly SIP allocation (₹)"

    plt.figure(figsize=(10, 5))

    if kind == "area":
        # Stacked area plot
        plt.stackplot(
            monthly.index,
            *[monthly[col].values for col in monthly.columns],
            labels=monthly.columns,
        )
    else:
        # Stacked bar plot
        monthly.plot(kind="bar", stacked=True)
        plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(y_label)
    plt.legend(title="Fund", loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    # Example manual test
    rotation_funds = [
        "SBI Multi Asset Allocation Fund - Direct Plan - Growth",
        "UTI Multi Asset Allocation Fund - Direct Plan - Growth Option",
        "HDFC Multi-Asset Fund - Direct Plan - Growth Option",
    ]

    baseline_fund = "SBI Multi Asset Allocation Fund - Direct Plan - Growth"

    comparison_df, rotation_res, baseline_res, normal_res, rotation_v2_res = compare_rotation_vs_baseline(
        rotation_funds=rotation_funds,
        baseline_fund=baseline_fund,
        sip_amount=100000.0,
        lambda_vol=0.5,
        lambda_dd=0.5,
        risk_free_rate=0.06,
    )

    print("Comparison Results:")
    print(comparison_df[["strategy_name", "total_invested", "final_value", "xirr", "volatility_annual", "max_drawdown"]])

 