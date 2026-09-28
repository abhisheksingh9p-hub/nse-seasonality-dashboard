"""
NSE Equity Seasonality Dashboard
"""

import datetime
import engine
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NSE Seasonality Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 NSE Equity Seasonality Dashboard")

st.sidebar.header("🔍 Filters & Options")

with st.sidebar.expander("📅 Date Range Settings", expanded=True):
    default_start = datetime.date(2026, 8, 1)
    default_end = datetime.date(2026, 8, 10)

    date_range = st.date_input(
        "Date Range",
        value=(default_start, default_end),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, default_end

    date_flexibility = st.slider("Date Flexibility (± Days)", 0, 5, 0)
    lookback_years = st.slider("Historical Lookback (Years)", 3, 10, 10)
    holding_days = st.slider("Holding Window (Cal. Days)", 10, 60, 20)

with st.sidebar.expander("🎯 Hard Filters", expanded=True):
    min_win_ratio = st.slider("Min Win Ratio (%)", 50, 100, 80)
    max_loss_limit = st.slider("Max Drawdown Limit (%)", -20, 0, -7)
    min_trades = st.slider("Min Required Years", 3, 10, 5)

if st.sidebar.button("🚀 Run Scan", type="primary", use_container_width=True):
    st.session_state["ran_scan"] = True

if st.session_state.get("ran_scan", False):
    with st.spinner("⚡ Running In-Memory Fast Scan..."):
        try:
            df_results = engine.run_seasonality_scan(
                start_date_obj=start_date,
                end_date_obj=end_date,
                holding_days=holding_days,
                lookback_years=lookback_years,
                min_win_ratio=min_win_ratio,
                max_loss_limit=max_loss_limit,
                date_flexibility_days=date_flexibility,
                min_trades=min_trades,
            )
            st.session_state["df_results"] = df_results
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

if "df_results" in st.session_state and not st.session_state["df_results"].empty:
    df_results = st.session_state["df_results"]

    st.success(f"Found {len(df_results)} matching seasonal patterns.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patterns Found", len(df_results))
    c2.metric("Avg Expected Return", f"{df_results['Avg Return (%)'].mean():.2f}%")
    c3.metric("Avg Expectancy", f"{df_results['Expectancy (%)'].mean():.2f}%")
    c4.metric("Avg Sharpe Ratio", f"{df_results['Sharpe Ratio'].mean():.2f}")

    st.subheader("📊 Seasonal Opportunities Table")
    st.caption("👈 **Row Selection Enabled:** Table me kisi bhi row par click karein us stock ka historical breakdown dekhne ke liye.")

    event = st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Win Ratio (%)": st.column_config.ProgressColumn("Win Ratio (%)", min_value=0, max_value=100, format="%.1f%%"),
            "Annualized Return (%)": st.column_config.NumberColumn("Annualized Return", format="%.2f%%"),
            "Last Close Price (₹)": st.column_config.NumberColumn("Last Trade Price (₹)", format="₹%.2f"),
        },
    )

    selected_symbol = None
    entry_date = None

    selected_rows = event.selection.get("rows", [])
    if selected_rows:
        row_idx = selected_rows[0]
        selected_symbol = df_results.iloc[row_idx]["Symbol"]
        entry_date = df_results.iloc[row_idx]["Entry Date"]

    st.divider()

    col_sel1, col_sel2 = st.columns([2, 4])
    with col_sel1:
        chosen_sym = st.selectbox(
            "🔍 Or Select Stock Symbol to View Breakdown:",
            options=df_results["Symbol"].unique(),
            index=list(df_results["Symbol"].unique()).index(selected_symbol) if selected_symbol else 0,
        )
        if not selected_symbol:
            selected_symbol = chosen_sym
            entry_date = df_results[df_results["Symbol"] == selected_symbol].iloc[0]["Entry Date"]

    if selected_symbol and entry_date:
        st.subheader(f"🔍 Historical Pattern Breakdown: **{selected_symbol}** ({lookback_years} Years Lookback)")
        st.caption(f"Seasonal Entry: **{entry_date}** | Holding Period: **{holding_days} Days**")

        df_breakdown = engine.get_symbol_yearly_breakdown(
            symbol=selected_symbol,
            entry_date_str=entry_date,
            holding_days=holding_days,
            lookback_years=lookback_years,
        )

        if not df_breakdown.empty:
            b_col1, b_col2 = st.columns([5, 5])

            with b_col1:
                st.markdown("##### Year-by-Year Seasonal Return Chart")
                fig = px.bar(
                    df_breakdown,
                    x="Year",
                    y="Return (%)",
                    color="Result",
                    color_discrete_map={"🟢 Profit": "#10B981", "🔴 Loss": "#EF4444"},
                    text="Return (%)",
                    title=f"{selected_symbol} - Return per Year for {entry_date} Window",
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(showlegend=True, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with b_col2:
                st.markdown("##### Year-by-Year Trade Log")
                st.dataframe(
                    df_breakdown,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
                    },
                )
        else:
            st.warning("No historical breakdown data available for this selection.")

elif st.session_state.get("ran_scan", False):
    st.warning("No patterns matched these exact filter conditions.")
else:
    st.info("Set your parameters in the sidebar and click **Run Scan**.")
