
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import rotation_engine

# Page config

st.set_page_config(
    page_title="InvestEdge – SIP Rotation Lab",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
}
h1, h2, h3 {
    font-weight: 600;
}
.metric-card {
    padding: 0.9rem 1.1rem;
    border-radius: 0.9rem;
    background: #020617;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.7);
}
.stButton > button {
    background-color: #22c55e;
    color: #020617;
    border-radius: 999px;
    border: none;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #16a34a;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
]

st.sidebar.title("InvestEdge Lab")
st.sidebar.caption("Configure your rotation experiment")

# ------------- Fund selection -------------
baseline_fund = st.sidebar.selectbox(
    "Baseline Fund (Benchmark):",
    options=ALL_FUNDS,
    index=5,
)

rotation_funds = st.sidebar.multiselect(
    "Rotation Universe (Select 2-4 Funds):",
    options=ALL_FUNDS,
    default=[ALL_FUNDS[0], ALL_FUNDS[1]],
    help="SIP will rotate only inside this set, with 10% minimum per fund.",
)

if len(rotation_funds) < 2:
    st.sidebar.warning("Select at least 2 funds for rotation.")
    run_disabled = True
elif len(rotation_funds) > 4:
    st.sidebar.warning("Rotation supports up to 4 funds.")
    run_disabled = True
else:
    run_disabled = False

# ------------- SIP & parameters -------------
sip_amount = st.sidebar.number_input(
    "Monthly SIP Amount (INR):",
    min_value=1000,
    max_value=5000000,
    value=100000,
    step=5000,
)

lambda_vol = st.sidebar.slider("Risk penalty (lambda_vol)", 0.0, 2.0, 0.5, 0.1)
lambda_dd = st.sidebar.slider("Drawdown penalty (lambda_dd)", 0.0, 2.0, 0.5, 0.1)
risk_free_rate = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 10.0, 6.0, 0.5) / 100.0

st.sidebar.caption("Start/end will default to common overlap of all funds.")
start_date = st.sidebar.date_input("Optional start date:", value=None)
end_date = st.sidebar.date_input("Optional end date:", value=None)

# Convert date_input None-like to real None or Timestamp
start_date_val = pd.to_datetime(start_date) if start_date else None
end_date_val = pd.to_datetime(end_date) if end_date else None

run_btn = st.sidebar.button("Run Analysis", disabled=run_disabled)

# Main header

st.markdown("## SIP Rotation Analysis")
st.markdown(
    "Compare a **data-driven rotation SIP** "
    "against a simple SIP into a single baseline fund."
)

# Explanation expander
with st.expander("Strategy Logic", expanded=False):
    st.write(
        """
        1. You pick 2–4 funds (ideally one per category).
        2. Every month, a fixed SIP amount is invested:
           - 10% goes to **each** selected fund (diversification floor).
           - The remaining SIP is allocated using a **score** based on:
             - Multi-horizon returns (1/3/6 month),
             - Penalised by 6-month volatility and 6-month max drawdown.
        3. Over time, the portfolio tilts more towards funds with stronger, more stable performance,
           while always keeping some allocation in every chosen fund.
        4. We compare this rotation strategy against a classic SIP in one fund using:
           XIRR and Excess Return vs Target (12%).
        """
    )

def format_currency(value: float) -> str:

    if value >= 10000000:
        return f"₹ {value / 10000000:.2f} Cr"
    elif value >= 100000:
        return f"₹ {value / 100000:.2f} L"
    elif value >= 1000:
        return f"₹ {value / 1000:.2f} K"
    else:
        return f"₹ {value:.2f}"

# Run backtest

if run_btn and not run_disabled:
    try:
        with st.spinner("Running backtest..."):
            comparison_df, rotation_res, baseline_res, normal_res, rotation_v2_res = rotation_engine.compare_rotation_vs_baseline(
                rotation_funds=rotation_funds,
                baseline_fund=baseline_fund,
                sip_amount=float(sip_amount),
                lambda_vol=lambda_vol,
                lambda_dd=lambda_dd,
                benchmark_name=baseline_fund,  # treat baseline as benchmark for alpha/beta
                risk_free_rate=risk_free_rate,
                start_date=start_date_val,
                end_date=end_date_val,
            )

  
        # KPI row
    
        st.markdown("### Key Performance Indicators")

        # Extract rows
        base_row = comparison_df[comparison_df["strategy_name"].str.startswith("Baseline")].iloc[0]
        rot_row = comparison_df[~comparison_df["strategy_name"].str.startswith("Baseline")].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "XIRR (Rotation)",
                f"{rot_row['xirr']*100:.2f} %",
                delta=f"{(rot_row['xirr'] - base_row['xirr'])*100:.2f} pp vs baseline",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Final Portfolio Value",
                format_currency(rot_row['final_value']),
                delta=f"{format_currency(rot_row['final_value'] - base_row['final_value'])} vs baseline",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Maximum Drawdown",
                f"{rot_row['max_drawdown']*100:.1f} %",
                delta=f"{(rot_row['max_drawdown'] - base_row['max_drawdown'])*100:.1f} pp vs baseline",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Full comparison table
        st.markdown("#### Comparative Performance Metrics")
        show_cols = [
            "strategy_name",
            "total_invested",
            "final_value",
            "absolute_return",
            "xirr",
            "num_installments",
        ]
        comp_to_show = comparison_df[show_cols].copy()
        
        # Apply formatting to currency columns for display
        comp_to_show["total_invested"] = comp_to_show["total_invested"].apply(lambda x: format_currency(x).replace("₹ ", ""))
        comp_to_show["final_value"] = comp_to_show["final_value"].apply(lambda x: format_currency(x).replace("₹ ", ""))

        comp_to_show["absolute_return"] = (comp_to_show["absolute_return"] * 100).round(2)
        comp_to_show["xirr"] = (comp_to_show["xirr"] * 100).round(2)

        
        comp_to_show["excess_return_12"] = ((comparison_df["xirr"] - 0.12) * 100).round(2)

        # Rename columns for display
        comp_to_show.rename(columns={
            "strategy_name": "Strategy",
            "total_invested": "Total Invested",
            "final_value": "Final Value",
            "absolute_return": "Absolute Return (%)",
            "xirr": "XIRR (%)",
            "excess_return_12": "Excess Return vs Target%",
            "num_installments": "Installments"
        }, inplace=True)

        st.dataframe(comp_to_show, use_container_width=True)

        st.markdown("---")

        # ==============================
        # Charts – tabs
        # ==============================
        tab1, tab2, tab3 = st.tabs(["Equity curve", "Drawdown", "Allocation history"])

        with tab1:

            rotation_engine.plot_equity_curves(baseline_res, rotation_res, normal_res, rotation_v2_res,
                               title="Baseline vs Normal vs Strategy 1 vs Strategy 2 – Portfolio Value")
            st.pyplot(plt.gcf())

        with tab2:
            rotation_engine.plot_drawdown_curves(baseline_res, rotation_res, normal_res, rotation_v2_res,
                                 title="Baseline vs Normal vs Strategy 1 vs Strategy 2 – Drawdown")
            st.pyplot(plt.gcf())

        with tab3:

            alloc_strategy = st.radio(
                "Select strategy to view allocation:",
                ["Rotation v1 (Score based)", "Rotation v2 (Performance + Dip)"],
                horizontal=True
            )
            
            if alloc_strategy == "Rotation v1 (Score based)":
                target_res = rotation_res
            else:
                target_res = rotation_v2_res

            if target_res.allocation_history is None or target_res.allocation_history.empty:
                st.info(f"No allocation history recorded for {target_res.name}.")
            else:
                rotation_engine.plot_allocation_history(
                    target_res,
                    as_weights=True,
                    kind="area",
                    title=f"Monthly SIP Allocation – {target_res.name}",
                )
                st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Backtest failed: {e}")
else:
    st.info("Select funds and parameters in the sidebar then click **Run Backtest**.")
