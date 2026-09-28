"""
NSE Equity Seasonality Dashboard
Speed: for each symbol, ALL (scan date x lookback year) trades are computed in
one numpy pass (searchsorted + small candidate matrix). No per-year Python loop,
no pandas filtering in the hot path.

Consistency: the scan and the per-symbol breakdown both go through
`_trade_arrays`, so the chart always reproduces the table row.

Rules (unchanged from v2):
  * Entry = trading day NEAREST the target date (within +-flex, up to +gap after)
  * Exit  = first trading day on/after entry + holding_days, at most `gap` late
  * Lookback = latest N calendar years whose full window fits in the dataset
  * NaN / non-positive prices rejected
  * Optional split/bonus guard (prices are unadjusted)
"""

from datetime import date, datetime, timedelta
from math import comb

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

PARQUET_PATH = (
    "https://huggingface.co/datasets/Abhishek9045/nse-bhavcopy-data/resolve/main/"
    "Master_EQ_Bhavcopy_Till_July.parquet"
)

MAX_GAP_DAYS = 5
MIN_HISTORY_ROWS = 200
STALE_DAYS = 30
SPLIT_RATIO = 0.6
RISK_FREE_PCT = 6.0

_EPOCH_ORD = date(1970, 1, 1).toordinal()


def _epoch_days(d):
    """python date -> days since 1970-01-01 (int)."""
    return d.toordinal() - _EPOCH_ORD


def _to_dt64(days):
    return np.datetime64(int(days), "D")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_resource(ttl=86400, show_spinner=False)
def load_master_data():
    """
    {"symbols": {sym: (days_int64, closes_float64, suspect_cumsum_int64)},
     "data_start": date, "data_end": date}
    """
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT Symbol, TradeDate, Close
        FROM '{PARQUET_PATH}'
        ORDER BY Symbol ASC, TradeDate ASC
        """
    ).df()
    con.close()

    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df = df.drop_duplicates(subset=["Symbol", "TradeDate"], keep="last")
    df = df.sort_values(["Symbol", "TradeDate"], kind="stable").reset_index(drop=True)

    all_days = df["TradeDate"].to_numpy().astype("datetime64[D]").astype(np.int64)
    all_close = df["Close"].to_numpy(dtype=np.float64)

    symbols = {}
    for sym, idx in df.groupby("Symbol", sort=False).indices.items():
        d = all_days[idx]
        c = all_close[idx]

        sus = np.zeros(len(c), dtype=np.int64)
        if len(c) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = c[1:] / c[:-1]
                sus[1:] = (ratio < SPLIT_RATIO) | (ratio > 1.0 / SPLIT_RATIO)
        symbols[sym] = (d, c, np.cumsum(sus))

    return {
        "symbols": symbols,
        "data_start": _to_dt64(all_days.min()).astype(object),
        "data_end": _to_dt64(all_days.max()).astype(object),
    }


# --------------------------------------------------------------------------- #
# Core vectorized trade logic (single source of truth)
# --------------------------------------------------------------------------- #
def candidate_years(data_start, data_end, month, day, holding_days, lookback_years):
    """Most recent `lookback_years` calendar years whose full window fits in the data."""
    years = []
    y = data_end.year
    while len(years) < lookback_years and y >= data_start.year:
        try:
            target = date(y, month, day)
        except ValueError:  # Feb 29 in a non-leap year
            y -= 1
            continue
        if target + timedelta(days=holding_days) <= data_end:
            years.append(y)
        y -= 1
    return years


def _trade_arrays(d, c, sus_cum, T, flex_days, holding_days, gap, split_guard):
    """
    d: int64 trading days (sorted), c: closes, sus_cum: suspect cumsum
    T: int64 array of target days (any length) -> vectorized over all of them.
    Returns (entry_idx, exit_idx, ret_pct) ; ret_pct is NaN where no valid trade.
    """
    n = len(d)
    lo = np.searchsorted(d, T - flex_days, side="left")
    hi = np.searchsorted(d, T + flex_days + gap, side="right")
    valid = hi > lo

    # candidate matrix: trading days inside [lo, hi) can't exceed W calendar days
    W = 2 * flex_days + gap + 1
    cand = lo[:, None] + np.arange(W)[None, :]
    inwin = cand < hi[:, None]
    diff = np.abs(d[np.minimum(cand, n - 1)] - T[:, None])
    diff = np.where(inwin, diff, np.iinfo(np.int64).max)
    i = np.minimum(lo + diff.argmin(axis=1), n - 1)   # nearest day, ties -> earlier

    X = d[i] + holding_days
    j = np.searchsorted(d, X, side="left")
    valid &= j < n
    jc = np.minimum(j, n - 1)
    valid &= d[jc] <= X + gap

    if split_guard:
        valid &= (sus_cum[jc] - sus_cum[i]) == 0

    ep = c[i]
    xp = c[jc]
    with np.errstate(invalid="ignore", divide="ignore"):
        valid &= (ep > 0) & (xp > 0)          # also rejects NaN
        ret = (xp / ep - 1.0) * 100.0
    return i, jc, np.where(valid, ret, np.nan)


def compute_trades(d, c, sus_cum, years, month, day, holding_days,
                   flex_days=0, max_gap=MAX_GAP_DAYS, split_guard=True):
    """Trade list for ONE calendar date (used by the breakdown).
    Returns [(year, entry_date, exit_date, entry_px, exit_px, ret_pct), ...]"""
    T, ys = [], []
    for y in years:
        try:
            T.append(_epoch_days(date(y, month, day)))
            ys.append(y)
        except ValueError:
            continue
    if not T:
        return []
    T = np.asarray(T, dtype=np.int64)
    i, j, ret = _trade_arrays(d, c, sus_cum, T, flex_days, holding_days, max_gap, split_guard)

    out = []
    for k in np.nonzero(~np.isnan(ret))[0]:
        out.append((ys[k], _to_dt64(d[i[k]]), _to_dt64(d[j[k]]),
                    float(c[i[k]]), float(c[j[k]]), float(ret[k])))
    return out


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def _binom_pvalue(wins, n):
    """P(X >= wins), X ~ Binomial(n, 0.5)."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / float(2 ** n)


def summarize(returns, holding_days, min_win_ratio, max_loss_limit, filter_skew):
    """Apply hard filters. Returns stats dict or None if rejected."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    wins = int(np.sum(r > 0))
    win_ratio = wins / n * 100.0
    if win_ratio < min_win_ratio:
        return None

    max_loss = float(np.min(r))
    if max_loss < max_loss_limit:
        return None

    avg = float(np.mean(r))
    med = float(np.median(r))
    if avg <= 0:
        return None
    if filter_skew and (med <= 0 or not (0.80 <= avg / med <= 1.25)):
        return None

    periods = 365.0 / holding_days
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    rf_per_trade = RISK_FREE_PCT / periods
    sharpe = ((avg - rf_per_trade) / std) * np.sqrt(periods) if std > 0 else 0.0

    return {
        "wins": wins, "n": n, "win_ratio": win_ratio,
        "avg": avg, "median": med,
        "annualized": avg * periods,          # simple, not compounded
        "sharpe": float(sharpe), "max_loss": max_loss,
        "pvalue": _binom_pvalue(wins, n),
    }


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def run_seasonality_scan(
    start_date_obj,
    end_date_obj,
    holding_days,
    lookback_years=10,
    min_win_ratio=80.0,
    max_loss_limit=-7.0,
    date_flexibility_days=0,
    min_trades=5,
    filter_skew=True,
    split_guard=True,
    progress_callback=None,
):
    if start_date_obj > end_date_obj:
        raise ValueError("Start date must be on or before end date.")

    data = load_master_data()
    symbols = data["symbols"]
    data_start, data_end = data["data_start"], data["data_end"]

    # ---- build ALL targets (scan date x year) once, outside the symbol loop ----
    scan_dates = list(pd.date_range(start=start_date_obj, end=end_date_obj, freq="D"))
    kept_dates, t_row, t_slot, t_day = [], [], [], []
    for d in scan_dates:
        years = candidate_years(data_start, data_end, d.month, d.day,
                                holding_days, lookback_years)
        if len(years) < min_trades:
            continue
        row = len(kept_dates)
        kept_dates.append(d)
        for slot, y in enumerate(years):
            t_row.append(row)
            t_slot.append(slot)
            t_day.append(_epoch_days(date(y, d.month, d.day)))

    if not kept_dates:
        return pd.DataFrame()

    t_row = np.asarray(t_row)
    t_slot = np.asarray(t_slot)
    T = np.asarray(t_day, dtype=np.int64)
    n_rows, n_slots = len(kept_dates), lookback_years
    labels = [d.strftime("%b %d") for d in kept_dates]

    stale_cutoff = _epoch_days(data_end - timedelta(days=STALE_DAYS))
    results = []
    total = len(symbols)

    for idx, (sym, (d, c, sus_cum)) in enumerate(symbols.items()):
        if progress_callback is not None and idx % 50 == 0:
            progress_callback(idx, total, sym)

        if len(d) < MIN_HISTORY_ROWS or d[-1] < stale_cutoff:
            continue

        # every (date, year) trade for this symbol in one shot
        _, _, ret = _trade_arrays(d, c, sus_cum, T, date_flexibility_days,
                                  holding_days, MAX_GAP_DAYS, split_guard)

        R = np.full((n_rows, n_slots), np.nan)
        R[t_row, t_slot] = ret

        ok = ~np.isnan(R)
        n_tr = ok.sum(axis=1)
        wins = (R > 0).sum(axis=1)

        # cheap vector pre-filter; `summarize` stays the authoritative filter
        cand = (n_tr >= min_trades) & (wins * 100.0 >= min_win_ratio * n_tr)
        rows = np.nonzero(cand)[0]
        if rows.size == 0:
            continue

        latest_price = float(c[-1])
        latest_date = str(_to_dt64(d[-1]))

        for k in rows:
            stats = summarize(R[k][ok[k]], holding_days, min_win_ratio,
                              max_loss_limit, filter_skew)
            if stats is None:
                continue
            results.append({
                "Symbol": sym,
                "Entry Date": labels[k],
                "Win Trades": f"{stats['wins']}/{stats['n']}",
                "Win Ratio (%)": round(stats["win_ratio"], 2),
                "P-value (vs coin flip)": round(stats["pvalue"], 4),
                "Avg Return (%)": round(stats["avg"], 2),
                "Median Return (%)": round(stats["median"], 2),
                "Annualized Return (%)": round(stats["annualized"], 2),
                "Sharpe Ratio": round(stats["sharpe"], 2),
                "Max Loss (%)": round(stats["max_loss"], 2),
                "Last Close Price (₹)": round(latest_price, 2),
                "Last Trade Date": latest_date,
            })

    if progress_callback is not None:
        progress_callback(total, total, "")

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # best date per symbol -> selection bias, stats are optimistic (see app caption)
        result_df = (
            result_df.sort_values(by=["Win Ratio (%)", "Avg Return (%)"], ascending=False)
            .drop_duplicates(subset=["Symbol"], keep="first")
            .sort_values(by="Annualized Return (%)", ascending=False)
            .reset_index(drop=True)
        )
    return result_df


# --------------------------------------------------------------------------- #
# Breakdown (same core function + same params as the scan)
# --------------------------------------------------------------------------- #
def get_symbol_yearly_breakdown(symbol, entry_date_str, holding_days,
                                lookback_years=10, date_flexibility_days=0,
                                split_guard=True):
    data = load_master_data()
    entry = data["symbols"].get(symbol)
    if entry is None:
        return pd.DataFrame()
    d, c, sus_cum = entry

    parsed = datetime.strptime(f"{entry_date_str} 2000", "%b %d %Y")  # 2000 = leap year
    years = candidate_years(data["data_start"], data["data_end"],
                            parsed.month, parsed.day, holding_days, lookback_years)

    trades = compute_trades(d, c, sus_cum, years, parsed.month, parsed.day,
                            holding_days, date_flexibility_days, MAX_GAP_DAYS, split_guard)
    if not trades:
        return pd.DataFrame()

    rows = [{
        "Year": y,
        "Entry Date": str(ed),
        "Exit Date": str(xd),
        "Entry Price (₹)": round(ep, 2),
        "Exit Price (₹)": round(xp, 2),
        "Return (%)": round(ret, 2),
        "Result": "🟢 Profit" if ret > 0 else "🔴 Loss",
    } for (y, ed, xd, ep, xp, ret) in trades]

    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
