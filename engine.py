"""
NSE Equity Seasonality Dashboard
"""

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

PARQUET_PATH = "https://huggingface.co/datasets/Abhishek9045/nse-bhavcopy-data/resolve/main/Master_EQ_Bhavcopy_Till_July.parquet"


@st.cache_data(ttl=86400, show_spinner=False)
def load_master_data():
    """Loads and caches the dataset in server RAM once."""
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT Symbol, TradeDate, Open, High, Low, Close
        FROM '{PARQUET_PATH}'
        ORDER BY Symbol ASC, TradeDate ASC
    """
    ).df()
    con.close()

    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["Year"] = df["TradeDate"].dt.year
    df["Month"] = df["TradeDate"].dt.month
    df["Day"] = df["TradeDate"].dt.day

    grouped = {sym: group.reset_index(drop=True) for sym, group in df.groupby("Symbol", sort=False)}
    return grouped


def run_seasonality_scan(
    start_date_obj,
    end_date_obj,
    holding_days,
    lookback_years=10,
    min_win_ratio=80.0,
    max_loss_limit=-7.0,
    date_flexibility_days=0,
    min_trades=5,
    progress_callback=None,
):
    if start_date_obj > end_date_obj:
        raise ValueError("Start date must be on or before end date.")

    # Fetch cached dataset directly from RAM
    grouped = load_master_data()
    symbols = list(grouped.keys())

    results = []
    scan_dates = pd.date_range(start=start_date_obj, end=end_date_obj, freq="D")
    total_symbols = len(symbols)

    for sym_idx, sym in enumerate(symbols):
        if progress_callback is not None and sym_idx % 20 == 0:
            progress_callback(sym_idx, total_symbols, sym)

        df = grouped[sym]
        if len(df) < 200:
            continue

        latest_row = df.iloc[-1]
        latest_price = float(latest_row["Close"])
        latest_price_date = latest_row["TradeDate"].strftime("%Y-%m-%d")

        all_years = sorted(df["Year"].unique())
        selected_years = set(all_years[-lookback_years:])
        if len(selected_years) < min_trades:
            continue

        # Fast Filter Symbol Data for relevant years
        df_sym = df[df["Year"].isin(selected_years)]

        for single_date in scan_dates:
            start_month = single_date.month
            start_day = single_date.day

            trades = []
            for y in selected_years:
                try:
                    entry_target = pd.Timestamp(year=y, month=start_month, day=start_day)
                except ValueError:
                    continue

                min_target = entry_target - pd.Timedelta(days=date_flexibility_days)
                max_target = entry_target + pd.Timedelta(days=date_flexibility_days + 5)

                entry_data = df_sym[(df_sym["TradeDate"] >= min_target) & (df_sym["TradeDate"] <= max_target)]
                if entry_data.empty:
                    continue

                entry_row = entry_data.iloc[0]
                actual_entry_date = entry_row["TradeDate"]
                exit_target = actual_entry_date + pd.Timedelta(days=holding_days)

                exit_data = df_sym[df_sym["TradeDate"] >= exit_target]
                if exit_data.empty:
                    continue

                exit_row = exit_data.iloc[0]
                entry_price = entry_row["Close"]
                exit_price = exit_row["Close"]

                if not entry_price:
                    continue

                trade_return = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append(trade_return)

            total_trades = len(trades)
            if total_trades < min_trades:
                continue

            returns_arr = np.array(trades)
            winning_trades = int(np.sum(returns_arr > 0))
            win_ratio = (winning_trades / total_trades) * 100.0

            if win_ratio < min_win_ratio:
                continue

            max_loss = float(np.min(returns_arr))
            if max_loss < max_loss_limit:
                continue

            avg_return = float(np.mean(returns_arr))
            median_return = float(np.median(returns_arr))

            if median_return == 0:
                continue
            mean_median_ratio = avg_return / median_return
            if not (0.80 <= mean_median_ratio <= 1.25):
                continue

            wins = returns_arr[returns_arr > 0]
            losses = np.abs(returns_arr[returns_arr <= 0])
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

            expectancy = ((win_ratio / 100.0) * avg_win) - (((100.0 - win_ratio) / 100.0) * avg_loss)
            if expectancy <= 0:
                continue

            periods_per_year = 365.0 / holding_days
            growth_factor = 1.0 + (avg_return / 100.0)
            annualized_return = (
                ((growth_factor ** periods_per_year) - 1.0) * 100.0 if growth_factor > 0 else -100.0
            )

            std_dev = float(np.std(returns_arr, ddof=1)) if total_trades > 1 else 0.0
            annualized_std = (std_dev / 100.0) * np.sqrt(periods_per_year) * 100.0
            risk_free_rate = 6.0
            sharpe_ratio = (
                (annualized_return - risk_free_rate) / annualized_std
                if (annualized_std > 0)
                else 0.0
            )

            results.append({
                "Symbol": sym,
                "Entry Date": single_date.strftime("%b %d"),
                "Win Trades": f"{winning_trades}/{total_trades}",
                "Win Ratio (%)": round(win_ratio, 2),
                "Avg Return (%)": round(avg_return, 2),
                "Median Return (%)": round(median_return, 2),
                "Annualized Return (%)": round(annualized_return, 2),
                "Sharpe Ratio": round(sharpe_ratio, 2),
                "Max Loss (%)": round(max_loss, 2),
                "Expectancy (%)": round(expectancy, 2),
                "Last Close Price (₹)": round(latest_price, 2),
                "Last Trade Date": latest_price_date,
            })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by="Annualized Return (%)", ascending=False)
        result_df = result_df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)

    return result_df


def get_symbol_yearly_breakdown(symbol, entry_date_str, holding_days, lookback_years=10):
    grouped = load_master_data()
    df = grouped.get(symbol)
    if df is None or df.empty:
        return pd.DataFrame()

    all_years = sorted(df["Year"].unique())
    selected_years = set(all_years[-lookback_years:])

    parsed_date = pd.to_datetime(entry_date_str, format="%b %d")
    start_month = parsed_date.month
    start_day = parsed_date.day

    yearly_details = []

    for y in selected_years:
        try:
            entry_target = pd.Timestamp(year=y, month=start_month, day=start_day)
        except ValueError:
            continue

        entry_data = df[df["TradeDate"] >= entry_target]
        if entry_data.empty:
            continue

        entry_row = entry_data.iloc[0]
        actual_entry_date = entry_row["TradeDate"]
        exit_target = actual_entry_date + pd.Timedelta(days=holding_days)

        exit_data = df[df["TradeDate"] >= exit_target]
        if exit_data.empty:
            continue

        exit_row = exit_data.iloc[0]
        entry_price = float(entry_row["Close"])
        exit_price = float(exit_row["Close"])

        trade_return = ((exit_price - entry_price) / entry_price) * 100.0

        yearly_details.append({
            "Year": y,
            "Entry Date": actual_entry_date.strftime("%Y-%m-%d"),
            "Exit Date": exit_row["TradeDate"].strftime("%Y-%m-%d"),
            "Entry Price (₹)": round(entry_price, 2),
            "Exit Price (₹)": round(exit_price, 2),
            "Return (%)": round(trade_return, 2),
            "Result": "🟢 Profit" if trade_return > 0 else "🔴 Loss",
        })

    return pd.DataFrame(yearly_details)
