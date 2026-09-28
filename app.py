"""
NSE Equity Seasonality Dashboard
Requires Streamlit >= 1.50 for width="stretch" (use use_container_width=True on older versions).
"""

import datetime

import engine
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NSE Seasonality Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 NSE Equity Seasonality Dashboard")

# Dataset freshness (cached after first load)
with st.spinner("Loading dataset..."):
    _meta = engine.load_master_data()
st.caption(f"Data coverage: {_meta['data_start']} → **{_meta['data_end']}**")

MAX_SCAN_DAYS = 60

# ------------------------------- Sidebar ---------------------------------- #
st.sidebar.header("🔍 Filters & Options")

with st.sidebar.expander("📅 Date Range Settings", expanded=True):
    today = datetime.date.today()
    date_range = st.date_input(
        "Entry Date Range",
        value=(today, today + datetime.timedelta(days=9)),
    )
    date_flexibility = st.slider("Date Flexibility (± Days)", 0, 5, 0)
    lookback_years = st.slider("Historical Lookback (Years)", 3, 10, 10)
    holding_days = st.slider("Holding Window (Cal. Days)", 10, 60, 20)

with st.sidebar.expander("🎯 Hard Filters", expanded=True):
    min_win_ratio = st.slider("Min Win Ratio (%)", 50, 100, 80)
    max_loss_limit = st.slider("Max Drawdown Limit (%)", -20, 0, -7)
    min_trades = st.slider("Min Required Years (trades)", 3, 10, 5)

with st.sidebar.expander("⚙️ Advanced", expanded=False):
    filter_skew = st.checkbox(
        "Drop outlier-driven patterns",
        value=True,
        help="Requires mean/median return ratio within 0.80–1.25 and a positive median.",
    )
    split_guard = st.checkbox(
        "Skip trades spanning a probable split/bonus",
        value=True,
        help="Prices are unadjusted. A one-day close move beyond ~-40% / +67% "
        "inside a trade window is treated as a corporate action and the trade is skipped.",
    )

# Validate the date input (a 1-tuple appears while the user is mid-selection)
valid_range = isinstance(date_range, (tuple, list)) and len(date_range) == 2
if valid_range:
    start_date, end_date = date_range
    span = (end_date - start_date).days + 1
    if start_date > end_date:
        st.sidebar.error("Start date must be on or before end date.")
        valid_range = False
    elif span > MAX_SCAN_DAYS:
        st.sidebar.error(f"Please keep the range to {MAX_SCAN_DAYS} days or fewer (currently {span}).")
        valid_range = False
else:
    st.sidebar.info("Pick both a start and an end date.")

run_clicked = st.sidebar.button(
    "🚀 Run Scan", type="primary", width="stretch", disabled=not valid_range
)

# ------------------------------- Run scan --------------------------------- #
# The scan runs ONLY when the button is pressed; results + the exact params
# used are stored so later reruns (row clicks etc.) never recompute or drift.
if run_clicked and valid_range:
    params = dict(
        holding_days=holding_days,
        lookback_years=lookback_years,
        date_flexibility_days=date_flexibility,
        split_guard=split_guard,
    )
    bar = st.progress(0.0, text="Scanning...")

    def _progress(i, total, sym):
        bar.progress(min(i / max(total, 1), 1.0), text=f"Scanning {sym} ({i}/{total})")

    try:
        df = engine.run_seasonality_scan(
            start_date_obj=start_date,
            end_date_obj=end_date,
            holding_days=holding_days,
            lookback_years=lookback_years,
            min_win_ratio=min_win_ratio,
            max_loss_limit=max_loss_limit,
            date_flexibility_days=date_flexibility,
            min_trades=min_trades,
            filter_skew=filter_skew,
            split_guard=split_guard,
            progress_callback=_progress,
        )
        st.session_state["scan"] = {"df": df, "params": params}
        st.session_state.pop("last_row", None)
        st.session_state.pop("sym_select", None)
    except Exception as e:
        bar.empty()
        st.exception(e)
        st.stop()
    bar.empty()

# ------------------------------- Results ---------------------------------- #
scan = st.session_state.get("scan")

if scan is None:
    st.info("Set your parameters in the sidebar and click **Run Scan**.")
    st.stop()

df_results = scan["df"]
p = scan["params"]

if df_results.empty:
    st.warning("No patterns matched these filter conditions.")
    st.stop()

st.success(f"Found {len(df_results)} matching seasonal patterns.")
st.caption(
    "⚠️ Exploratory only. With ~2,000 stocks × several dates, many 8/10 win records occur by chance — "
    "check the **P-value** column (lower is better), and note each symbol shows its best-scoring date "
    "(selection bias). Prices are unadjusted; no costs, slippage or dividends. "
    "Annualized return is simple (avg × 365 / holding days)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patterns Found", len(df_results))
c2.metric("Avg Return / Trade", f"{df_results['Avg Return (%)'].mean():.2f}%")
c3.metric("Avg Median Return", f"{df_results['Median Return (%)'].mean():.2f}%")
c4.metric("Avg Sharpe Ratio", f"{df_results['Sharpe Ratio'].mean():.2f}")

st.subheader("📊 Seasonal Opportunities Table")
st.caption("👈 Click a row to see that stock's year-by-year breakdown.")

event = st.dataframe(
    df_results,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="results_table",
    column_config={
        "Win Ratio (%)": st.column_config.ProgressColumn(
            "Win Ratio (%)", min_value=0, max_value=100, format="%.1f%%"
        ),
        "Annualized Return (%)": st.column_config.NumberColumn("Annualized Return", format="%.2f%%"),
        "P-value (vs coin flip)": st.column_config.NumberColumn("P-value", format="%.4f"),
        "Last Close Price (₹)": st.column_config.NumberColumn("Last Close (₹)", format="₹%.2f"),
    },
)

# A newly clicked row updates the selectbox; the selectbox can then override it.
rows = event.selection.get("rows", [])
row_now = rows[0] if rows else None
if row_now is not None and row_now != st.session_state.get("last_row"):
    st.session_state["sym_select"] = df_results.iloc[row_now]["Symbol"]
st.session_state["last_row"] = row_now

st.divider()

symbols = list(df_results["Symbol"])
if st.session_state.get("sym_select") not in symbols:
    st.session_state["sym_select"] = symbols[0]

col_sel, _ = st.columns([2, 4])
with col_sel:
    selected_symbol = st.selectbox("🔍 Stock to inspect:", options=symbols, key="sym_select")

entry_date = df_results.loc[df_results["Symbol"] == selected_symbol, "Entry Date"].iloc[0]

st.subheader(f"🔍 Historical Pattern Breakdown: **{selected_symbol}** ({p['lookback_years']} Years Lookback)")
st.caption(
    f"Seasonal entry: **{entry_date}** | Holding: **{p['holding_days']} days** | "
    f"Flexibility: **±{p['date_flexibility_days']}d** (parameters used for this scan)"
)

df_breakdown = engine.get_symbol_yearly_breakdown(
    symbol=selected_symbol,
    entry_date_str=entry_date,
    holding_days=p["holding_days"],
    lookback_years=p["lookback_years"],
    date_flexibility_days=p["date_flexibility_days"],
    split_guard=p["split_guard"],
)

if df_breakdown.empty:
    st.warning("No historical breakdown data available for this selection.")
else:
    b1, b2 = st.columns(2)

    with b1:
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
        fig.update_xaxes(type="category")
        fig.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig, width="stretch")

    with b2:
        st.markdown("##### Year-by-Year Trade Log")
        st.dataframe(
            df_breakdown,
            width="stretch",
            hide_index=True,
            column_config={
                "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
            },
        )
