from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from returns_engine import (
    load_clean_nav,
    compute_volatility,
    compute_max_drawdown,
    compute_sortino_ratio,
    compute_alpha_beta,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean"
PLOTS_DIR = DATA_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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


def run_baseline_sip_strategy(
    fund_name: str,
    sip_amount: float = 10000.0,
    frequency: str = "M",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
    fund_nav = load_clean_nav(fund_name).df.sort_values("date")

    if start_date is None:
        start_date = fund_nav["date"].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = fund_nav["date"].max()
    else:
        end_date = pd.to_datetime(end_date)

    df_period = fund_nav[(fund_nav["date"] >= start_date) & (fund_nav["date"] <= end_date)].copy()
    if df_period.empty:
        raise ValueError("No NAV data in the chosen period for baseline strategy.")

    sip_dates = pd.date_range(start=start_date, end=end_date, freq=frequency).to_pydatetime().tolist()
    sip_records = []
    for d in sip_dates:
        nav_row = df_period[df_period["date"] >= pd.to_datetime(d)]
        if nav_row.empty:
            continue
        nav_date = nav_row["date"].iloc[0]
        nav_price = float(nav_row["nav"].iloc[0])
        idx = nav_row.index[0]
        units = sip_amount / nav_price
        sip_records.append((idx, nav_date, sip_amount, units))

    if not sip_records:
        raise ValueError("No SIP trades executed for baseline strategy.")

    df_port = df_period[["date", "nav"]].copy()
    df_port["units_delta"] = 0.0

    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []

    for idx, nav_date, amount, units in sip_records:
        df_port.loc[idx, "units_delta"] += units
        cashflows.append(-amount)
        cashflow_dates.append(nav_date)

    df_port["units"] = df_port["units_delta"].cumsum()
    df_port["nav_portfolio"] = df_port["units"] * df_port["nav"]
    portfolio_df = df_port[["date", "nav_portfolio"]].rename(columns={"nav_portfolio": "nav"})

    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0
    xirr = _compute_xirr(cashflows, cashflow_dates)

    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    vol = compute_volatility(portfolio_df)
    mdd, peak, trough = compute_max_drawdown(portfolio_df)
    sortino = compute_sortino_ratio(portfolio_df, risk_free_rate=risk_free_rate)

    alpha_annual = None
    beta = None
    if benchmark_name is not None:
        bench_df = load_clean_nav(benchmark_name).df
        bench_df = bench_df[(bench_df["date"] >= portfolio_df["date"].min()) &
                            (bench_df["date"] <= portfolio_df["date"].max())].copy()
        if not bench_df.empty:
            ab = compute_alpha_beta(portfolio_df, bench_df, risk_free_rate=risk_free_rate)
            alpha_annual = ab.get("alpha_annual")
            beta = ab.get("beta")

    return StrategyResult(
        name=f"Baseline SIP in {fund_name}",
        description=f"Fixed {sip_amount} {frequency} SIP into {fund_name}.",
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=len([cf for cf in cashflows if cf < 0]),
        start_date=portfolio_df["date"].min(),
        end_date=portfolio_df["date"].max(),
        volatility_annual=vol,
        max_drawdown=mdd,
        max_dd_peak_date=peak,
        max_dd_trough_date=trough,
        sortino_ratio=sortino,
        alpha_annual=alpha_annual,
        beta=beta,
        portfolio_nav=portfolio_df,
    )


def _load_multi_fund_nav(
    fund_names: List[str],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    nav_series = {}
    for name in fund_names:
        df = load_clean_nav(name).df[["date", "nav"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]
        df = df.sort_values("date").set_index("date")
        nav_series[name] = df["nav"]

    combined = pd.concat(nav_series, axis=1)
    combined.columns = fund_names
    combined = combined.sort_index().ffill()
    combined = combined.dropna(how="all")
    return combined


def run_momentum_sip_strategy(
    fund_names: List[str],
    sip_amount: float = 10000.0,
    frequency: str = "M",
    lookback_days: int = 126,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
    if len(fund_names) < 2:
        raise ValueError("Momentum strategy needs at least two funds to choose from.")

    temp_nav = load_clean_nav(fund_names[0]).df
    temp_nav["date"] = pd.to_datetime(temp_nav["date"])

    if start_date is None:
        start_date = temp_nav["date"].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = temp_nav["date"].max()
    else:
        end_date = pd.to_datetime(end_date)

    combined_nav = _load_multi_fund_nav(fund_names, start_date, end_date)
    if combined_nav.empty:
        raise ValueError("No NAV data available for momentum strategy period.")

    sip_dates = pd.date_range(start=start_date, end=end_date, freq=frequency).to_pydatetime().tolist()

    units_delta = {name: pd.Series(0.0, index=combined_nav.index) for name in fund_names}

    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []
    actual_sip_count = 0

    for d in sip_dates:
        current_date = pd.to_datetime(d)
        possible_dates = combined_nav.index[combined_nav.index >= current_date]
        if len(possible_dates) == 0:
            continue
        nav_date = possible_dates[0]

        lb_start = nav_date - pd.Timedelta(days=lookback_days)
        hist_window = combined_nav.loc[(combined_nav.index > lb_start) & (combined_nav.index <= nav_date)].copy()

        if hist_window.empty:
            chosen_fund = fund_names[0]
        else:
            start_vals = hist_window.iloc[0]
            end_vals = hist_window.iloc[-1]
            trailing_ret = end_vals / start_vals - 1.0
            chosen_fund = trailing_ret.idxmax()

        nav_price = float(combined_nav.loc[nav_date, chosen_fund])
        units = sip_amount / nav_price
        units_delta[chosen_fund].loc[nav_date:] += units

        cashflows.append(-sip_amount)
        cashflow_dates.append(nav_date)
        actual_sip_count += 1

    if actual_sip_count == 0:
        raise ValueError("No SIP trades executed in momentum strategy.")

    portfolio_values = pd.Series(0.0, index=combined_nav.index)
    for name in fund_names:
        portfolio_values += units_delta[name] * combined_nav[name]

    portfolio_df = pd.DataFrame({"date": portfolio_values.index, "nav": portfolio_values.values})

    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0
    xirr = _compute_xirr(cashflows, cashflow_dates)

    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    vol = compute_volatility(portfolio_df)
    mdd, peak, trough = compute_max_drawdown(portfolio_df)
    sortino = compute_sortino_ratio(portfolio_df, risk_free_rate=risk_free_rate)

    alpha_annual = None
    beta = None
    if benchmark_name is not None:
        bench_df = load_clean_nav(benchmark_name).df
        bench_df = bench_df[(bench_df["date"] >= portfolio_df["date"].min()) &
                            (bench_df["date"] <= portfolio_df["date"].max())].copy()
        if not bench_df.empty:
            ab = compute_alpha_beta(portfolio_df, bench_df, risk_free_rate=risk_free_rate)
            alpha_annual = ab.get("alpha_annual")
            beta = ab.get("beta")

    return StrategyResult(
        name="Momentum SIP Strategy",
        description=(
            f"SIP {sip_amount} {frequency}, each installment into the fund "
            f"with highest trailing return over last {lookback_days} days "
            f"among: {', '.join(fund_names)}."
        ),
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=actual_sip_count,
        start_date=portfolio_df["date"].min(),
        end_date=portfolio_df["date"].max(),
        volatility_annual=vol,
        max_drawdown=mdd,
        max_dd_peak_date=peak,
        max_dd_trough_date=trough,
        sortino_ratio=sortino,
        alpha_annual=alpha_annual,
        beta=beta,
        portfolio_nav=portfolio_df,
    )



def run_rotation_sip_strategy(
    fund_names: List[str],
    sip_amount: float = 100000.0,
    floor_weight: float = 0.10,
    frequency: str = "M",
    lookback_days: int = 126,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    benchmark_name: Optional[str] = None,
    risk_free_rate: float = 0.0,
) -> StrategyResult:
    """
    Rotation SIP with a minimum floor allocation to every fund.

    Rules:
      - User pre-selects 2–4 funds (different categories).
      - Every SIP date, at least `floor_weight` of the SIP amount goes to *each* fund.
      - Remaining amount is allocated to the fund with highest trailing return
        over the last `lookback_days` days among the chosen funds.
      - Resulting allocations always sum to 100% for that month.

    Example: 2 funds, floor_weight=0.10 -> each gets 10%, leftover 80% goes to
    the best performer for that lookback window.
    """

    if len(fund_names) < 2:
        raise ValueError("Rotation strategy needs at least two funds.")
    if floor_weight <= 0 or floor_weight >= 1:
        raise ValueError("floor_weight must be between 0 and 1.")
    if floor_weight * len(fund_names) >= 1.0:
        raise ValueError(
            f"floor_weight * num_funds = {floor_weight * len(fund_names):.2f} "
            "must be < 1 so that some capital is left to rotate."
        )

    # Use the first fund just to infer available date range
    temp_nav = load_clean_nav(fund_names[0]).df
    temp_nav["date"] = pd.to_datetime(temp_nav["date"])

    if start_date is None:
        start_date = temp_nav["date"].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = temp_nav["date"].max()
    else:
        end_date = pd.to_datetime(end_date)

    # Combined NAV matrix: index = dates, columns = funds
    combined_nav = _load_multi_fund_nav(fund_names, start_date, end_date)
    if combined_nav.empty:
        raise ValueError("No NAV data available for rotation strategy period.")

    # SIP schedule
    sip_dates = pd.date_range(start=start_date, end=end_date, freq=frequency).to_pydatetime().tolist()

    # Track monthly unit changes for each fund
    units_delta = {name: pd.Series(0.0, index=combined_nav.index) for name in fund_names}

    # Cashflows for XIRR calculation
    cashflows: List[float] = []
    cashflow_dates: List[pd.Timestamp] = []
    actual_sip_count = 0

    n_funds = len(fund_names)
    base_weight = floor_weight
    leftover_weight = 1.0 - base_weight * n_funds

    for d in sip_dates:
        current_date = pd.to_datetime(d)
        possible_dates = combined_nav.index[combined_nav.index >= current_date]
        if len(possible_dates) == 0:
            continue  # no NAV after this date
        nav_date = possible_dates[0]

        # Trailing window for ranking
        lb_start = nav_date - pd.Timedelta(days=lookback_days)
        hist_window = combined_nav.loc[(combined_nav.index > lb_start) & (combined_nav.index <= nav_date)].copy()

        if hist_window.empty:
            # If we don't have a history window yet, just split equally
            chosen_fund = fund_names[0]
        else:
            start_vals = hist_window.iloc[0]
            end_vals = hist_window.iloc[-1]
            trailing_ret = end_vals / start_vals - 1.0
            chosen_fund = trailing_ret.idxmax()

        # Build weights: base floor for all, leftover to best fund
        weights = {name: base_weight for name in fund_names}
        weights[chosen_fund] += leftover_weight

        # Sanity: small numerical drift fix
        total_w = sum(weights.values())
        for k in weights:
            weights[k] /= total_w

        # Apply SIP allocation according to weights
        for name in fund_names:
            nav_price = float(combined_nav.loc[nav_date, name])
            invest_amount = sip_amount * weights[name]
            units = invest_amount / nav_price

            units_delta[name].loc[nav_date] += units

        cashflows.append(-sip_amount)
        cashflow_dates.append(nav_date)
        actual_sip_count += 1

    if actual_sip_count == 0:
        raise ValueError("No SIP trades executed for rotation strategy.")

    # Build units-held series per fund and total portfolio NAV
    units_df = pd.DataFrame(
        {name: s.cumsum() for name, s in units_delta.items()},
        index=combined_nav.index,
    )
    portfolio_values = (units_df * combined_nav).sum(axis=1)

    portfolio_df = pd.DataFrame({"date": portfolio_values.index, "nav": portfolio_values.values})

    # Final value and XIRR
    final_value = float(portfolio_df["nav"].iloc[-1])
    cashflows.append(final_value)
    cashflow_dates.append(portfolio_df["date"].iloc[-1])

    total_invested = float(sum(-cf for cf in cashflows if cf < 0))
    abs_return = final_value / total_invested - 1.0
    xirr = _compute_xirr(cashflows, cashflow_dates)

    # Risk metrics
    portfolio_df = _ensure_portfolio_daily_return(portfolio_df)

    vol = compute_volatility(portfolio_df)
    mdd, peak, trough = compute_max_drawdown(portfolio_df)
    sortino = compute_sortino_ratio(portfolio_df, risk_free_rate=risk_free_rate)

    alpha_annual = None
    beta = None
    if benchmark_name is not None:
        bench_df = load_clean_nav(benchmark_name).df
        bench_df = bench_df[
            (bench_df["date"] >= portfolio_df["date"].min())
            & (bench_df["date"] <= portfolio_df["date"].max())
        ].copy()
        if not bench_df.empty:
            ab = compute_alpha_beta(portfolio_df, bench_df, risk_free_rate=risk_free_rate)
            alpha_annual = ab.get("alpha_annual")
            beta = ab.get("beta")

    return StrategyResult(
        name="Rotation SIP Strategy (10% floor per fund)",
        description=(
            f"SIP {sip_amount} {frequency} across {', '.join(fund_names)} with "
            f"{floor_weight:.0%} minimum to each fund and remaining allocated "
            f"to the best {lookback_days}-day performer."
        ),
        total_invested=total_invested,
        final_value=final_value,
        absolute_return=abs_return,
        xirr=xirr,
        num_installments=actual_sip_count,
        start_date=portfolio_df["date"].min(),
        end_date=portfolio_df["date"].max(),
        volatility_annual=vol,
        max_drawdown=mdd,
        max_dd_peak_date=peak,
        max_dd_trough_date=trough,
        sortino_ratio=sortino,
        alpha_annual=alpha_annual,
        beta=beta,
        portfolio_nav=portfolio_df,
    )


def compare_strategies_to_df(results: List[StrategyResult]) -> pd.DataFrame:
    rows = []
    for r in results:
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
    return pd.DataFrame(rows)


def plot_strategy_nav(
    strategy: StrategyResult,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Plot the portfolio NAV curve and save to disk.
    """
    if output_path is None:
        safe_name = strategy.name.lower().replace(" ", "_")
        output_path = PLOTS_DIR / f"{safe_name}_nav.png"

    plt.figure(figsize=(10, 5))
    plt.plot(strategy.portfolio_nav["date"], strategy.portfolio_nav["nav"], label=strategy.name)
    plt.title(title or f"Portfolio NAV – {strategy.name}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return output_path


def plot_strategy_comparison_bar(
    results: List[StrategyResult],
    metric: str = "final_value",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create a bar chart comparing a metric (final_value, xirr, etc.) across strategies.
    """
    if not results:
        raise ValueError("No strategy results provided for plotting.")

    valid_metrics = {
        "final_value": "Final Value",
        "total_invested": "Total Invested",
        "absolute_return": "Absolute Return",
        "xirr": "XIRR",
        "volatility_annual": "Annualised Volatility",
        "max_drawdown": "Max Drawdown",
        "sortino_ratio": "Sortino Ratio",
    }
    if metric not in valid_metrics:
        raise ValueError(f"Metric '{metric}' not supported. Choose from {list(valid_metrics.keys())}.")

    labels = [r.name for r in results]
    values = [getattr(r, metric) for r in results]

    if output_path is None:
        output_path = PLOTS_DIR / f"strategy_compare_{metric}.png"

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][: len(results)])
    plt.title(title or f"Strategy Comparison – {valid_metrics[metric]}")
    plt.ylabel(valid_metrics[metric])
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ax = plt.gca()
    is_percent_metric = metric in {"absolute_return", "xirr", "max_drawdown", "volatility_annual"}
    if is_percent_metric:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2%}"))
    for bar, value in zip(bars, values):
        label = f"{value:.2%}" if is_percent_metric else f"{value:,.0f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return output_path


if __name__ == "__main__":
    # Example experiment: Large Cap baseline vs Large+Small rotation
    baseline_fund = "SBI Large Cap Fund - Direct Plan - Growth"
    rotation_funds = [
        "SBI Large Cap Fund - Direct Plan - Growth",
        "SBI Small Cap Fund - Direct Plan - Growth",
    ]

    sip_amount = 100000.0
    rf = 0.06  # risk-free for alpha/Sortino
    lookback = 126  # ~6 months for ranking

    print("\n[Experiment] Running baseline SIP...")
    baseline = run_baseline_sip_strategy(
        fund_name=baseline_fund,
        sip_amount=sip_amount,
        frequency="M",
        risk_free_rate=rf,
    )

    print("[Experiment] Running rotation SIP strategy...")
    rotation = run_rotation_sip_strategy(
        fund_names=rotation_funds,
        sip_amount=sip_amount,
        floor_weight=0.10,
        frequency="M",
        lookback_days=lookback,
        risk_free_rate=rf,
        benchmark_name=baseline_fund,  # optional: treat Large Cap as benchmark
    )

    # Tabular comparison
    comp = compare_strategies_to_df([baseline, rotation])
    cols = [
        "strategy_name",
        "total_invested",
        "final_value",
        "absolute_return",
        "xirr",
        "volatility_annual",
        "max_drawdown",
        "sortino_ratio",
        "alpha_annual",
        "beta",
    ]
    print("\n=== Strategy Comparison ===")
    print(comp[cols].to_string(index=False))

    # Plots
    nav_plot_baseline = plot_strategy_nav(
        baseline,
        title=f"Baseline SIP – {baseline_fund}",
        output_path=PLOTS_DIR / "baseline_largecap_rotation_nav.png",
    )
    nav_plot_rotation = plot_strategy_nav(
        rotation,
        title="Rotation SIP Strategy (10% floor per fund)",
        output_path=PLOTS_DIR / "rotation_largecap_smallcap_nav.png",
    )
    comparison_final = plot_strategy_comparison_bar(
        [baseline, rotation],
        metric="final_value",
        title="Strategy Comparison – Final Portfolio Value",
        output_path=PLOTS_DIR / "rotation_vs_baseline_final.png",
    )
    comparison_xirr = plot_strategy_comparison_bar(
        [baseline, rotation],
        metric="xirr",
        title="Strategy Comparison – XIRR",
        output_path=PLOTS_DIR / "rotation_vs_baseline_xirr.png",
    )

    print(f"\nSaved plots:")
    print(f" - Baseline NAV: {nav_plot_baseline}")
    print(f" - Rotation NAV: {nav_plot_rotation}")
    print(f" - Final value comparison: {comparison_final}")
    print(f" - XIRR comparison: {comparison_xirr}")

