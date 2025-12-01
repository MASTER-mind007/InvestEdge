# signals.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from returns_engine import (
    load_clean_nav,
    compute_volatility,
    compute_max_drawdown,
)


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class FundSignals:
    fund_name: str
    df: pd.DataFrame  # columns: ['fund_name','date','nav','R1','R3','R6','VOL6','DD6']


def _compute_single_fund_signals(
    fund_name: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    min_window_days: int = 30,
) -> FundSignals:
    
    nav_obj = load_clean_nav(fund_name)
    df = nav_obj.df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Optional date filtering
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["date"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["date"] <= end_date]

    if df.empty:
        raise ValueError(f"No NAV data available for fund '{fund_name}' in the chosen period.")

        # 1) Monthly nav series (month-end) and simple 1/3/6M returns
        
    monthly_nav = (
        df.set_index("date")["nav"]
        .resample("M")
        .last()
        .dropna()
    )

    monthly_df = monthly_nav.to_frame(name="nav").reset_index()  # columns: ['date', 'nav']

    # 1M / 3M / 6M simple returns (price-based, not log returns)
    monthly_df["R1"] = monthly_df["nav"].pct_change(1)
    monthly_df["R3"] = monthly_df["nav"].pct_change(3)
    monthly_df["R6"] = monthly_df["nav"].pct_change(6)

    
    # 2) 6-month risk metrics using DAILY data: VOL6 (annualised) & DD6
    
    vol_list = []
    dd_list = []

    # ensure daily frame is sorted
    daily_df = df[["date", "nav"]].copy().sort_values("date")

    for dt_val in monthly_df["date"]:
        # 6 calendar months lookback
        window_start = dt_val - pd.DateOffset(months=6)
        window_df = daily_df[(daily_df["date"] > window_start) & (daily_df["date"] <= dt_val)].copy()

        if window_df.shape[0] < min_window_days:
            # not enough data to compute stable risk metrics
            vol_list.append(np.nan)
            dd_list.append(np.nan)
            continue

        try:
            vol6 = compute_volatility(window_df, annualize=True)
        except Exception:
            vol6 = np.nan

        try:
            # compute_max_drawdown expects columns ['date','nav']
            max_dd, _, _ = compute_max_drawdown(window_df)
        except Exception:
            max_dd = np.nan

        vol_list.append(vol6)
        dd_list.append(max_dd)

    monthly_df["VOL6"] = vol_list
    monthly_df["DD6"] = dd_list

    # Add fund_name
    monthly_df["fund_name"] = fund_name

    # Reorder columns for convenience
    cols = ["fund_name", "date", "nav", "R1", "R3", "R6", "VOL6", "DD6"]
    monthly_df = monthly_df[cols]

    return FundSignals(fund_name=fund_name, df=monthly_df)


def compute_signals_for_funds(
    fund_names: List[str],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    min_window_days: int = 30,
) -> pd.DataFrame:
    
    all_frames: List[pd.DataFrame] = []

    for name in fund_names:
        fs = _compute_single_fund_signals(
            fund_name=name,
            start_date=start_date,
            end_date=end_date,
            min_window_days=min_window_days,
        )
        all_frames.append(fs.df)

    if not all_frames:
        raise ValueError("No signals could be computed (empty fund_names or no data).")

    signals_df = pd.concat(all_frames, axis=0, ignore_index=True)
    signals_df = signals_df.sort_values(["date", "fund_name"]).reset_index(drop=True)
    return signals_df


if __name__ == "__main__":
    # Simple manual test hook (you can run: python signals.py)
    example_funds = [
        "SBI Multi Asset Allocation Fund - Direct Plan - Growth",
        "UTI Multi Asset Allocation Fund - Direct Plan - Growth Option",
    ]
    sig = compute_signals_for_funds(example_funds)
    print(sig.head())
