"""
NSE Seasonality Engine (Single Best Entry per Symbol Fix)
"""

import duckdb
import numpy as np
import pandas as pd

PARQUET_PATH = "https://huggingface.co/datasets/Abhishek9045/nse-bhavcopy-data/resolve/main/Master_EQ_Bhavcopy_Till_July.parquet"


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

    con = duckdb.connect()

    symbols = con.execute(
        f"""
        SELECT Symbol
        FROM '{PARQUET_PATH}'
        GROUP BY Symbol
        HAVING COUNT(*) > 500
        ORDER BY Symbol ASC
    """
    ).df()["Symbol"].tolist()

    symbols_sql_list = ", ".join(f"'{s}'" for s in symbols)
    full_df = con.execute(
        f"""
        SELECT Symbol, TradeDate, Open, High, Low, Close, Volume
        FROM '{PARQUET_PATH}'
        WHERE Symbol IN ({symbols_sql_list})
        ORDER BY Symbol ASC, TradeDate ASC
    """
    ).df()
    con.close()

    full_df["TradeDate"] = pd.to_datetime(full_df["TradeDate"])
    grouped = dict(tuple(full_df.groupby("Symbol", sort=False)))

    results = []
    scan_dates = pd.date_range(start=start_date_obj, end=end_date_obj, freq="D")
    total_symbols = len(symbols)

    for sym_idx, sym in enumerate(symbols):
        if progress_callback is not None:
            progress_callback(sym_idx, total_symbols, sym)

        df = grouped.get(sym)
        if df is None or len(df) < 300:
            continue

        df = df.reset_index(drop=True)
        df["Year"] = df["TradeDate"].dt.year

        latest_row = df.iloc[-1]
        latest_price = float(latest_row["Close"])
        latest_price_date = latest_row["TradeDate"].strftime("%Y-%m-%d")

        all_years = sorted(df["Year"].unique())
        selected_years = all_years[-lookback_years:]
        if len(selected_years) < 3:
            continue

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

                entry_data = df[(df["TradeDate"] >= min_target) & (df["TradeDate"] <= max_target)]
                if entry_data.empty:
                    continue

                entry_row = entry_data.iloc[0]
                actual_entry_date = entry_row["TradeDate"]
                exit_target = actual_entry_date + pd.Timedelta(days=holding_days)

                exit_data = df[df["TradeDate"] >= exit_target]
                if exit_data.empty:
                    continue

                exit_row = exit_data.iloc[0]
                entry_price = entry_row["Close"]
                exit_price = exit_row["Close"]

                if entry_price is None or entry_price == 0 or pd.isna(entry_price):
                    continue

                trade_return = ((exit_price - entry_price) / entry_price) * 100
                trades.append({"Year": y, "Return": trade_return})

            if not trades:
                continue

            tdf = pd.DataFrame(trades)
            total_trades = len(tdf)
            if total_trades < min_trades:
                continue

            winning_trades = int((tdf["Return"] > 0).sum())
            win_ratio = (winning_trades / total_trades) * 100

            wins = tdf.loc[tdf["Return"] > 0, "Return"]
            losses = tdf.loc[tdf["Return"] <= 0, "Return"]
            avg_win = wins.mean() if not wins.empty else 0
            avg_loss = abs(losses.mean()) if not losses.empty else 0

            avg_return = tdf["Return"].mean()
            median_return = tdf["Return"].median()
            max_loss = tdf["Return"].min()
            std_dev = tdf["Return"].std(ddof=1) if total_trades > 1 else 0.0

            expectancy = ((win_ratio / 100) * avg_win) - (((100 - win_ratio) / 100) * avg_loss)

            periods_per_year = 365.0 / holding_days
            growth_factor = 1 + (avg_return / 100)
            annualized_return = (
                ((growth_factor ** periods_per_year) - 1) * 100 if growth_factor > 0 else -100.0
            )

            annualized_std = (std_dev / 100.0) * np.sqrt(periods_per_year) * 100.0
            risk_free_rate = 6.0
            sharpe_ratio = (
                (annualized_return - risk_free_rate) / annualized_std
                if (not np.isnan(annualized_std) and annualized_std > 0)
                else 0.0
            )

            if median_return == 0:
                continue
            mean_median_ratio = avg_return / median_return

            if win_ratio < min_win_ratio or max_loss < max_loss_limit:
                continue
            if not (0.80 <= mean_median_ratio <= 1.25):
                continue
            if expectancy <= 0:
                continue

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
        # Highest Annualized Return se pehle sort karo
        result_df = result_df.sort_values(
            by="Annualized Return (%)", ascending=False
        )
        # --- DEDUPLICATION: Har Symbol ki sirf Pehli (Best) Row rakho ---
        result_df = result_df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)

    return result_df


def get_symbol_yearly_breakdown(symbol, entry_date_str, holding_days, lookback_years=10):
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT TradeDate, Open, High, Low, Close
        FROM '{PARQUET_PATH}'
        WHERE Symbol = '{symbol}'
        ORDER BY TradeDate ASC
    """
    ).df()
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["Year"] = df["TradeDate"].dt.year

    all_years = sorted(df["Year"].unique())
    selected_years = all_years[-lookback_years:]

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

        trade_return = ((exit_price - entry_price) / entry_price) * 100

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