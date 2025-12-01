# InvestEdge: SIP Rotation Lab

## Overview
InvestEdge is a quantitative backtesting framework designed to analyze and compare Systematic Investment Plan (SIP) strategies. It focuses on evaluating a "Rotation" strategy against traditional baseline SIPs. The system allows users to simulate portfolio performance using historical Net Asset Value (NAV) data, applying dynamic allocation rules based on multi-horizon returns and risk metrics.

## Features
- **Multi-Strategy Comparison**: Compare "Baseline SIP", "Equal Weight SIP", "Rotation v1 (Score-based)", and "Rotation v2 (Trend + Dip)".
- **Dynamic Allocation**: Simulate monthly rebalancing based on momentum (R1, R3, R6), volatility (VOL6), and drawdown (DD6) signals.
- **Interactive Dashboard**: A Streamlit-based UI for configuring backtest parameters (funds, dates, risk penalties) and visualizing results.
- **Comprehensive Metrics**: Calculate XIRR, Absolute Return, Volatility, Max Drawdown, Sortino Ratio, and Alpha/Beta.
- **Visualizations**: Interactive charts for Equity Curves, Drawdown Curves, and Monthly Allocation History.

## Installation

1.  **Clone the repository** (or extract the project files).
2.  **Install dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages:
    ```bash
    pip install pandas numpy matplotlib streamlit
    ```

## Usage

### Running the Dashboard
The primary interface is the Streamlit web application. To launch it, run:

```bash
streamlit run app_streamlit.py
```

This will open the dashboard in your default web browser.

### Configuration
1.  **Baseline Fund**: Select a benchmark fund for comparison.
2.  **Rotation Universe**: Choose 2-4 funds to include in the rotation strategy.
3.  **SIP Parameters**: Set the monthly investment amount, risk penalties, and backtest duration.
4.  **Run Analysis**: Click the button to execute the backtest and generate reports.

## Project Structure

- `app_streamlit.py`: Main entry point for the interactive dashboard.
- `rotation_engine.py`: Core logic for rotation strategies and backtesting execution.
- `returns_engine.py`: Library for financial calculations (CAGR, XIRR, Volatility, Drawdown).
- `signals.py`: Module for computing monthly trading signals (Momentum, Risk).
- `data/`: Directory containing historical NAV data (clean CSVs).

## Strategy Logic

### Rotation v1 (Score-based)
Allocates a fixed 10% floor to each selected fund. The remaining capital is distributed based on a composite score derived from:
- **Reward**: Weighted average of 1-month, 3-month, and 6-month returns.
- **Risk**: Penalties for 6-month volatility and maximum drawdown.

### Rotation v2 (Trend + Dip)
- **Core**: 50% of capital is allocated equally across all funds.
- **Trend**: 25% is allocated based on relative momentum strength.
- **Dip**: 25% is allocated to funds experiencing a drawdown, provided they have a positive long-term trend ("Buy the Dip").

## License
This project is for educational and research purposes only.
