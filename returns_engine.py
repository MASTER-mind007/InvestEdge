from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean"


@dataclass
class FundNav:
    fund_name: str
    df: pd.DataFrame


def _safe_name_from_fund(fund_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in fund_name.lower()).strip("_")


def _guess_clean_file_path(fund_name: str) -> Path:
    safe_name = _safe_name_from_fund(fund_name)
    return CLEAN_DATA_DIR / f"{safe_name}_clean.csv"


def load_clean_nav(fund_name: str, path: Optional[Path] = None) -> FundNav:
    """
    Load clean NAV data for a given fund.

    Args:
        fund_name: Name of the fund.
        path: Optional explicit path to the CSV file. If None, guesses based on fund name.

    Returns:
        FundNav object containing the fund name and DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        path = _guess_clean_file_path(fund_name)

    if not path.exists():
        raise FileNotFoundError(f"Clean NAV file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns or "nav" not in df.columns:
        raise ValueError(f"Expected 'date' and 'nav' columns in {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return FundNav(fund_name=fund_name, df=df)


def compute_cagr(
    start_value: float,
    end_value: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).

    Args:
        start_value: Initial investment value.
        end_value: Final investment value.
        start_date: Investment start date.
        end_date: Investment end date.

    Returns:
        CAGR as a decimal (e.g., 0.15 for 15%).
    """
    if start_value <= 0:
        raise ValueError("start_value must be > 0 for CAGR")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    days = (end_date - start_date).days
    years = days / 365.25
    return (end_value / start_value) ** (1 / years) - 1.0


def _ensure_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    if "daily_return" not in df.columns:
        df["nav_prev"] = df["nav"].shift(1)
        df["daily_return"] = df["nav"] / df["nav_prev"] - 1
        df.loc[df["nav_prev"].isna(), "daily_return"] = np.nan
    return df


def lumpsum_return(
    nav_df: pd.DataFrame,
    invest_date: pd.Timestamp,
    redemption_date: Optional[pd.Timestamp] = None,
    amount: float = 100000.0,
) -> Dict[str, float]:
    """
    Calculate returns for a lumpsum investment.

    Args:
        nav_df: DataFrame with 'date' and 'nav'.
        invest_date: Date of investment.
        redemption_date: Date of redemption (defaults to last available date).
        amount: Amount invested.

    Returns:
        Dictionary with initial_investment, final_value, absolute_return, cagr.
    """
    df = nav_df.copy().sort_values("date")

    invest_date = pd.to_datetime(invest_date)
    if redemption_date is None:
        redemption_date = df["date"].max()
    redemption_date = pd.to_datetime(redemption_date)

    df_invest = df[df["date"] >= invest_date]
    df_redeem = df[df["date"] <= redemption_date]

    if df_invest.empty or df_redeem.empty:
        raise ValueError("No NAV data available for the given date range.")

    nav_invest = df_invest["nav"].iloc[0]
    nav_redeem = df_redeem["nav"].iloc[-1]

    units = amount / nav_invest
    final_value = units * nav_redeem
    abs_return = final_value / amount - 1.0

    cagr = compute_cagr(
        start_value=amount,
        end_value=final_value,
        start_date=df_invest["date"].iloc[0],
        end_date=df_redeem["date"].iloc[-1],
    )

    return {
        "initial_investment": float(amount),
        "final_value": float(final_value),
        "absolute_return": float(abs_return),
        "cagr": float(cagr),
    }


def sip_return(
    nav_df: pd.DataFrame,
    sip_amount: float = 10000.0,
    frequency: str = "M",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    Calculate returns for a Systematic Investment Plan (SIP).

    Args:
        nav_df: DataFrame with 'date' and 'nav'.
        sip_amount: Amount invested per installment.
        frequency: SIP frequency ('M' for monthly).
        start_date: Start date of SIP.
        end_date: End date of SIP.

    Returns:
        Dictionary with total_invested, final_value, absolute_return, xirr, etc.
    """
    df = nav_df.copy().sort_values("date")

    if start_date is None:
        start_date = df["date"].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = df["date"].max()
    else:
        end_date = pd.to_datetime(end_date)

    df_period = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if df_period.empty:
        raise ValueError("No NAV data in the SIP period.")

    sip_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    sip_dates = sip_dates.to_pydatetime().tolist()

    sip_records = []
    for d in sip_dates:
        nav_row = df_period[df_period["date"] >= pd.to_datetime(d)]
        if nav_row.empty:
            continue
        nav_price = nav_row["nav"].iloc[0]
        nav_date = nav_row["date"].iloc[0]
        units = sip_amount / nav_price
        sip_records.append((nav_date, sip_amount, units))

    if not sip_records:
        raise ValueError("No SIP investments could be made (no matching NAV dates).")

    sip_df = pd.DataFrame(sip_records, columns=["date", "amount", "units"])

    total_invested = float(sip_df["amount"].sum())
    total_units = float(sip_df["units"].sum())

    last_nav_row = df_period[df_period["date"] <= end_date]
    if last_nav_row.empty:
        last_nav_row = df_period.tail(1)
    final_nav = float(last_nav_row["nav"].iloc[-1])
    final_value = total_units * final_nav

    abs_return = final_value / total_invested - 1.0

    cashflows = []
    dates = []
    for _, row in sip_df.iterrows():
        cashflows.append(-float(row["amount"]))
        dates.append(row["date"])

    cashflows.append(final_value)
    dates.append(last_nav_row["date"].iloc[-1])

    t0 = dates[0]
    year_fracs = np.array([(d - t0).days / 365.25 for d in dates])
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
    xirr = (low + high) / 2

    return {
        "total_invested": total_invested,
        "final_value": float(final_value),
        "absolute_return": float(abs_return),
        "xirr": float(xirr),
        "num_installments": int(len(sip_df)),
        "start_date": sip_df["date"].min(),
        "end_date": last_nav_row["date"].iloc[-1],
    }


def compute_volatility(
    nav_df: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """
    Compute annualized volatility of daily returns.

    Args:
        nav_df: DataFrame with 'date' and 'nav'.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days in a year (default 252).

    Returns:
        Volatility as a decimal.
    """
    df = _ensure_daily_return(nav_df)
    daily_ret = df["daily_return"].dropna()
    if daily_ret.empty:
        raise ValueError("No daily returns available to compute volatility.")

    daily_vol = float(daily_ret.std(ddof=1))
    if not annualize:
        return daily_vol
    return daily_vol * np.sqrt(trading_days)


def compute_max_drawdown(nav_df: pd.DataFrame) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Compute Maximum Drawdown (MDD).

    Args:
        nav_df: DataFrame with 'date' and 'nav'.

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
        max_drawdown is a negative float (e.g., -0.20 for 20% drop).
    """
    df = nav_df.copy().sort_values("date").reset_index(drop=True)
    nav = df["nav"].values
    cum_max = np.maximum.accumulate(nav)
    drawdowns = nav / cum_max - 1.0

    idx_trough = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[idx_trough])

    idx_peak = int(np.argmax(nav[: idx_trough + 1]))

    peak_date = df["date"].iloc[idx_peak]
    trough_date = df["date"].iloc[idx_trough]

    return max_dd, peak_date, trough_date


def _align_fund_and_benchmark(
    fund_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    f = _ensure_daily_return(fund_df)
    b = _ensure_daily_return(bench_df)

    f = f[["date", "daily_return"]].dropna().rename(columns={"daily_return": "fund_ret"})
    b = b[["date", "daily_return"]].dropna().rename(columns={"daily_return": "bench_ret"})

    merged = pd.merge(f, b, on="date", how="inner").sort_values("date")
    if merged.empty:
        raise ValueError("No overlapping dates between fund and benchmark for return alignment.")

    return merged["fund_ret"], merged["bench_ret"]


def compute_alpha_beta(
    fund_nav_df: pd.DataFrame,
    bench_nav_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute CAPM Alpha and Beta relative to a benchmark.

    Args:
        fund_nav_df: Fund NAV DataFrame.
        bench_nav_df: Benchmark NAV DataFrame.
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Dictionary with beta, alpha_daily, alpha_annual, etc.
    """
    fund_ret, bench_ret = _align_fund_and_benchmark(fund_nav_df, bench_nav_df)

    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    fund_exc = fund_ret - rf_daily
    bench_exc = bench_ret - rf_daily

    mask = fund_exc.notna() & bench_exc.notna()
    fund_exc = fund_exc[mask]
    bench_exc = bench_exc[mask]

    if len(fund_exc) < 2:
        raise ValueError("Not enough overlapping return observations to compute alpha/beta.")

    cov = np.cov(fund_exc, bench_exc, ddof=1)[0, 1]
    var_bench = np.var(bench_exc, ddof=1)
    if var_bench == 0:
        raise ValueError("Benchmark variance is zero; cannot compute beta.")
    beta = cov / var_bench

    mean_fund_exc = float(fund_exc.mean())
    mean_bench_exc = float(bench_exc.mean())

    alpha_daily = mean_fund_exc - beta * mean_bench_exc
    alpha_annual = (1 + alpha_daily) ** trading_days - 1

    return {
        "beta": float(beta),
        "alpha_daily": float(alpha_daily),
        "alpha_annual": float(alpha_annual),
        "mean_excess_fund_daily": mean_fund_exc,
        "mean_excess_bench_daily": mean_bench_exc,
    }


def compute_sortino_ratio(
    fund_nav_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute Sortino Ratio (annualized).

    Args:
        fund_nav_df: DataFrame with 'date' and 'nav'.
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Sortino ratio. Returns NaN if no downside deviation.
    """
    df = _ensure_daily_return(fund_nav_df)
    ret = df["daily_return"].dropna()
    if ret.empty:
        raise ValueError("No daily returns available to compute Sortino ratio.")

    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = ret - rf_daily

    mean_excess_daily = float(excess.mean())

    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")

    downside_std_daily = float(downside.std(ddof=1))

    annualised_excess = mean_excess_daily * trading_days
    annualised_downside = downside_std_daily * np.sqrt(trading_days)

    if annualised_downside == 0:
        return float("nan")

    sortino = annualised_excess / annualised_downside
    return float(sortino)


def summarise_fund_metrics(
    fund_name: str,
    sip_amount: float = 10000.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, object]:
    fund_nav = load_clean_nav(fund_name)
    df = fund_nav.df

    ls = lumpsum_return(df, invest_date=df["date"].min())
    sip = sip_return(df, sip_amount=sip_amount, frequency="M")
    vol = compute_volatility(df)
    mdd, peak, trough = compute_max_drawdown(df)
    sortino = compute_sortino_ratio(df, risk_free_rate=risk_free_rate)

    return {
        "fund_name": fund_name,
        "lumpsum": ls,
        "sip": sip,
        "volatility_annual": vol,
        "max_drawdown": mdd,
        "max_drawdown_peak_date": peak,
        "max_drawdown_trough_date": trough,
        "sortino_ratio": sortino,
    }


if __name__ == "__main__":
    example_fund = "SBI Multi Asset Allocation Fund - Direct Plan - Growth"
    metrics = summarise_fund_metrics(example_fund, sip_amount=10000.0, risk_free_rate=0.06)
    print(metrics)

