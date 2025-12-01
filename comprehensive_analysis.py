
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from data_backbone import load_fund_configs, CLEAN_DATA_DIR
from returns_engine import load_clean_nav, compute_volatility, compute_cagr, _ensure_daily_return

PLOTS_DIR = Path(__file__).resolve().parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_comprehensive_analysis():
    print("Loading fund configurations...")
    funds = load_fund_configs()
    
    fund_metrics = []
    all_navs = {}

    print(f"Processing {len(funds)} funds...")
    for fund in funds:
        try:
            # Load data
            fund_nav = load_clean_nav(fund.fund_name)
            df = fund_nav.df
            
            if df.empty:
                continue

            # Calculate Metrics
            # CAGR
            start_val = df["nav"].iloc[0]
            end_val = df["nav"].iloc[-1]
            start_date = df["date"].iloc[0]
            end_date = df["date"].iloc[-1]
            years = (end_date - start_date).days / 365.25
            
            if years < 1:
                cagr = (end_val / start_val - 1) * (1/years) # Simple annualization for < 1 year
            else:
                cagr = (end_val / start_val) ** (1 / years) - 1.0

            # Volatility
            vol = compute_volatility(df)

            # Max Drawdown
            nav = df["nav"].values
            cum_max = np.maximum.accumulate(nav)
            drawdowns = nav / cum_max - 1.0
            max_dd = drawdowns.min()

            fund_metrics.append({
                "fund_name": fund.fund_name,
                "category": fund.category,
                "cagr": cagr,
                "volatility": vol,
                "max_drawdown": max_dd,
                "years": years
            })

            # Store for correlation/NAV plots (resampled to daily to align)
            df = df.set_index("date")
            all_navs[fund.fund_name] = df["nav"]

        except Exception as e:
            print(f"Skipping {fund.fund_name}: {e}")

    metrics_df = pd.DataFrame(fund_metrics)
    
    # 1. Risk-Return Scatter Plot
    plot_risk_return(metrics_df)

    # 2. Category Performance Bar Chart
    plot_category_performance(metrics_df)

    # 3. Correlation Heatmap (Representative funds)
    plot_correlation_heatmap(all_navs, metrics_df)

    # 4. Category Winners NAV
    plot_category_winners(all_navs, metrics_df)

def plot_risk_return(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="volatility", y="cagr", hue="category", style="category", s=100)
    
    # Label points
    for i, row in df.iterrows():
        # Only label if interesting (e.g. high return or high risk) or sparse
        if row["cagr"] > 0.20 or row["volatility"] > 0.20: 
             plt.text(row["volatility"]+0.002, row["cagr"], row["fund_name"][:20], fontsize=8, alpha=0.7)

    plt.title("Risk vs Return Profile (All Funds)")
    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("CAGR (Return)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "risk_return_scatter.png", dpi=150)
    print("Saved risk_return_scatter.png")
    plt.close()

def plot_category_performance(df):
    cat_grp = df.groupby("category")[["cagr", "max_drawdown"]].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # CAGR Bar
    sns.barplot(data=cat_grp, x="category", y="cagr", ax=ax1, color="skyblue", alpha=0.8, label="Avg CAGR")
    ax1.set_ylabel("Average CAGR")
    ax1.set_title("Average Performance by Category")
    ax1.tick_params(axis='x', rotation=45)

    # Drawdown Line (Secondary Axis)
    ax2 = ax1.twinx()
    sns.lineplot(data=cat_grp, x="category", y="max_drawdown", ax=ax2, color="red", marker="o", label="Avg Max Drawdown")
    ax2.set_ylabel("Average Max Drawdown")
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_performance.png", dpi=150)
    print("Saved category_performance.png")
    plt.close()

def plot_correlation_heatmap(all_navs, metrics_df):
    # Select one fund per category to keep heatmap readable
    rep_funds = metrics_df.sort_values("cagr", ascending=False).groupby("category").head(1)["fund_name"].tolist()
    
    combined_df = pd.DataFrame({name: all_navs[name] for name in rep_funds if name in all_navs})
    combined_df = combined_df.ffill().dropna()
    
    # Calculate daily returns for correlation
    corr_matrix = combined_df.pct_change().corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Top Fund per Category)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    print("Saved correlation_heatmap.png")
    plt.close()

def plot_category_winners(all_navs, metrics_df):
    # Identify top fund in each category by CAGR
    winners = metrics_df.loc[metrics_df.groupby("category")["cagr"].idxmax()]
    
    plt.figure(figsize=(12, 7))
    
    for _, row in winners.iterrows():
        name = row["fund_name"]
        cat = row["category"]
        if name in all_navs:
            nav_series = all_navs[name]
            # Normalize to 100
            normalized = (nav_series / nav_series.iloc[0]) * 100
            plt.plot(normalized.index, normalized.values, label=f"{cat}: {name[:20]}...")

    plt.title("Growth of ₹100 - Top Performer by Category")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_winners_nav.png", dpi=150)
    print("Saved category_winners_nav.png")
    plt.close()

if __name__ == "__main__":
    run_comprehensive_analysis()
