# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
Alpha Vantage as data source (time-series + overview)
Author: Biswanath Das (ported)
Notes:
 - Put your ALPHA_VANTAGE_KEY below.
 - Alpha Vantage free tier limits: 5 requests/minute — caching is used to reduce calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from email.message import EmailMessage
import smtplib
from datetime import datetime

# -------------------------
# Config / API key
# -------------------------
ALPHA_VANTAGE_KEY = "4VGPC0X8LHDXAIIC"  # <-- replace with your key or load from env
AV_BASE = "https://www.alphavantage.co/query"
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
MOCK_SLEEP = 0.02

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="StockMentor (AlphaVantage)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (Alpha Vantage)")
st.caption("Alpha Vantage for price data + overview where available. Use .BSE / .NS etc. API limits apply.")

# -------------------------
# Utilities: Alpha Vantage requests with safe parsing
# -------------------------
def av_call(params: dict):
    """Call Alpha Vantage with given params. Returns parsed json (or {})."""
    params = dict(params)
    params["apikey"] = ALPHA_VANTAGE_KEY
    try:
        resp = requests.get(AV_BASE, params=params, timeout=20)
        data = resp.json()
        # Alpha Vantage rate-limit / note keys often under 'Note' or 'Error Message'
        if not data:
            return {}
        return data
    except Exception as e:
        print("AV call error:", e)
        return {}

@st.cache_data(ttl=900)
def fetch_time_series_daily(symbol: str, outputsize: str = "compact"):
    """
    Return pandas DataFrame of daily OHLCV indexed by Date for symbol.
    symbol examples: 'RELIANCE.BSE' or 'TCS.NS'
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
    }
    data = av_call(params)
    key = "Time Series (Daily)"
    if key not in data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data[key], orient="index")
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=900)
def fetch_overview(symbol: str):
    """Return the Overview dict for the symbol (may be empty for some markets)."""
    params = {"function": "OVERVIEW", "symbol": symbol}
    data = av_call(params)
    # If empty dict or missing typical fields, return {}
    if not data:
        return {}
    return data

# -------------------------
# Watchlist utilities
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        symbols = [s.replace('.NS', '').strip().upper() for s in symbols if s and str(s).strip()]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading {WATCHLIST_FILE}: {e}")
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        try:
            load_watchlist.clear()  # clear cache
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Helpers: convert symbol formats
# -------------------------
def normalize_for_av(sym_no_suffix: str):
    """
    Convert a symbol like 'RELIANCE' or 'TCS' to a likely AlphaVantage symbol:
    prefer .BSE (many Indian examples) or .NS — user may already include suffix.
    Strategy:
     - if input contains '.' keep as-is (user gave 'TCS.NS' or 'RELIANCE.BSE')
     - else append '.BSE' (you may change to .NS if you prefer)
    """
    s = str(sym_no_suffix).strip().upper()
    if "." in s:
        return s
    # default: BSE (common), you may prefer .NS — change here if needed
    return s + ".BSE"

# -------------------------
# Safe extraction helpers (map AV overview keys to app usage)
# -------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def overview_get_numeric(ov, keys):
    """Try multiple keys to get numeric value from overview JSON."""
    for k in keys:
        if k in ov and ov[k] not in (None, "", "None"):
            return safe_float(ov[k])
    return np.nan

# -------------------------
# Price CAGR helper (Alpha Vantage time-series)
# -------------------------
def price_cagr_from_df(df: pd.DataFrame, years: int = 5):
    try:
        if df is None or df.empty:
            return None
        df = df.sort_index()
        latest = df.index.max()
        start_date = latest - pd.DateOffset(years=years)
        df_period = df[df.index >= start_date]
        if df_period.empty or len(df_period) < 10:
            # not enough points in target window
            # fallback: find price approx years ago by nearest available date
            earliest = df.index.min()
            total_days = (latest - earliest).days
            if total_days < 365 * (years - 0.5):
                return None
            # approximate: get price at earliest and latest and compute CAGR
            start_price = df["close"].iloc[0]
            end_price = df["close"].iloc[-1]
            n = (len(df_period) - 1) if len(df_period) > 1 else years
        else:
            start_price = df_period["close"].iloc[0]
            end_price = df_period["close"].iloc[-1]
            n = years
        if start_price <= 0 or end_price <= 0 or n <= 0:
            return None
        cagr = ((end_price / start_price) ** (1.0 / n) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return None

# -------------------------
# Convert Alpha Vantage overview -> info-like dict used by rest of app
# -------------------------
def build_info_from_overview(symbol_av: str):
    """
    Returns an `info` dict with keys used across app (currentPrice, trailingPE, bookValue, dividendYield, heldPercentInsiders, returnOnEquity, debtToEquity, marketCap, beta)
    Many of these may be missing for Indian tickers; functions will handle fallbacks.
    """
    ov = fetch_overview(symbol_av)
    info = {}
    # market cap
    mc = overview_get_numeric(ov, ["MarketCapitalization", "MarketCapitalization"])
    info["marketCap"] = mc if not np.isnan(mc) else None
    # PE
    pe = overview_get_numeric(ov, ["PERatio", "PERatio"])
    info["trailingPE"] = pe if not np.isnan(pe) else None
    peg = overview_get_numeric(ov, ["PEGRatio", "PEGRatio"])
    info["pegRatio"] = peg if not np.isnan(peg) else None
    # book value
    bv = overview_get_numeric(ov, ["BookValue", "BookValue"])
    info["bookValue"] = bv if not np.isnan(bv) else None
    # dividend yield (ALPHA may give DividendYield as e.g. "0.5" meaning percent or ratio; try to parse safely)
    dy = None
    if "DividendYield" in ov:
        try:
            dy_raw = ov["DividendYield"]
            # some providers give "0.01" (ratio) or "1.0" (percent) — attempt to normalize:
            dyf = float(dy_raw)
            if dyf > 1:  # looks like percent (e.g., 2.5)
                dy = round(dyf / 100.0, 6)  # store as ratio
            else:
                dy = round(dyf, 6)
        except Exception:
            dy = None
    info["dividendYield"] = dy
    # return on equity TTM
    roe_keys = ["ReturnOnEquityTTM", "ReturnOnEquityTTM"]
    roe = overview_get_numeric(ov, roe_keys)
    if not np.isnan(roe):
        # assume it's already in percent (e.g., "15.23") -> convert to fractional form like yfinance (0.15)
        info["returnOnEquity"] = roe / 100.0
    else:
        info["returnOnEquity"] = None
    # debt to equity: AV may not have; check keys
    de = overview_get_numeric(ov, ["DebtToEquity", "TotalDebt", "TotalDebt/Equity"])
    info["debtToEquity"] = de if not np.isnan(de) else None
    # insiders/promoter holding (try multiple keys)
    promoter = overview_get_numeric(ov, ["InsiderPercent", "InsiderOwnership", "InstitutionalPercent"])
    if not np.isnan(promoter):
        info["heldPercentInsiders"] = promoter / 100.0
    else:
        info["heldPercentInsiders"] = None
    # beta not always available
    beta = overview_get_numeric(ov, ["Beta", "Beta"])
    info["beta"] = beta if not np.isnan(beta) else None
    # enterprise to ebitda not in overview, skip
    # currentPrice: not in overview -> we will compute from latest close in time series
    return info

# -------------------------
# Financial metrics helper using Alpha Vantage (overview + price fallback)
# -------------------------
@st.cache_data(ttl=900)
def get_financial_metrics_av(symbol_no_suffix: str):
    """
    For a given watch symbol (without .NS) return:
      - revenue/profit cagr (via price proxy, since fundamentals often unavailable)
      - roe_pct, debt_to_equity, dividend_yield_pct, promoter_holding_pct
    """
    symbol_av = normalize_for_av(symbol_no_suffix)
    out = {
        "revenue_cagr_pct": None,
        "profit_cagr_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "dividend_yield_pct": None,
        "promoter_holding_pct": None,
    }
    # 1) try Overview for fundamentals
    ov = fetch_overview(symbol_av)
    if ov:
        # ROE
        roe = None
        for k in ("ReturnOnEquityTTM", "ReturnOnEquity"):
            if k in ov and ov[k] not in (None, ""):
                try:
                    roe = float(ov[k])
                    break
                except Exception:
                    pass
        if roe is not None:
            out["roe_pct"] = round(roe, 2)
        # Debt-to-equity (if present)
        for k in ("DebtToEquity", "DebtToEquityTTM"):
            if k in ov and ov[k] not in (None, ""):
                try:
                    out["debt_to_equity"] = float(ov[k])
                    break
                except Exception:
                    pass
        # dividend yield
        if "DividendYield" in ov and ov["DividendYield"] not in (None, ""):
            try:
                dy = float(ov["DividendYield"])
                # normalization: if >1 likely percent
                if dy > 1:
                    dy = dy / 100.0
                out["dividend_yield_pct"] = round(dy * 100.0, 2)
            except Exception:
                pass
        # promoter/insider holding
        for k in ("InsiderPercent", "InsiderOwnership", "InsiderTransaction"):
            if k in ov and ov[k] not in (None, ""):
                try:
                    ph = float(ov[k])
                    if ph > 1:  # percent value -> convert to absolute percent
                        out["promoter_holding_pct"] = round(ph, 2)
                    else:
                        out["promoter_holding_pct"] = round(ph * 100.0, 2)
                    break
                except Exception:
                    pass

    # 2) Price-based CAGR as fallback / proxy for revenue/profit growth
    # try 3-year and 5-year price CAGRs
    df5 = fetch_time_series_daily(symbol_av, outputsize="full")
    if not df5.empty:
        cagr5 = price_cagr_from_df(df5, years=5)
        cagr3 = price_cagr_from_df(df5, years=3)
        # assign revenue_cagr -> prefer 3y if available, else 5y
        out["revenue_cagr_pct"] = cagr3 if cagr3 is not None else cagr5
        out["profit_cagr_pct"] = cagr3 if cagr3 is not None else cagr5

    return out

# -------------------------
# Estimate fair value and buy/sell zones using info dict built from overview + price EPS fallback
# -------------------------
def estimate_fair_value_av(info, symbol_av=None):
    """
    Use overview values (PERatio, BookValue) and price-based EPS fallback (not ideal).
    Returns (fair_value, method)
    """
    try:
        # Try analyst target price if present in overview? AV doesn't provide targetMeanPrice.
        # Try trailing PE from overview
        pe_target = DEFAULT_PE_TARGET
        if info.get("trailingPE") and info.get("trailingPE") > 0:
            pe_target = info["trailingPE"]
        # EPS is not provided by overview; attempt to approximate EPS from price and PERatio if both present:
        current_price = info.get("currentPrice", None)
        pe = info.get("trailingPE", None)
        if current_price and pe and pe > 0:
            eps_est = current_price / pe
            fv = eps_est * pe_target
            return round(float(fv), 2), f"EstEPSxPE({pe_target:.1f})"
        # fallback: book value * trailing PE if present
        if info.get("bookValue") and info.get("trailingPE"):
            fv = info["bookValue"] * info["trailingPE"]
            return round(float(fv), 2), "BVxPE"
        return None, "InsufficientData"
    except Exception:
        return None, "InsufficientData"

def compute_buy_sell(fair_value, mos=0.30):
    if fair_value is None or (isinstance(fair_value, float) and math.isnan(fair_value)):
        return None, None
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos / 1.5), 2)

# -------------------------
# Rule-based recommendation (reuse previous logic, using AV info)
# -------------------------
def rule_based_recommendation_av(info, fair_value, current_price, revenue_cagr=None, profit_cagr=None):
    # same logic as your previous function, slight name differences
    score = 0
    reasons = []

    roe = info.get("returnOnEquity", np.nan)
    roe_pct = None
    if isinstance(roe, (int, float)):
        try:
            if abs(roe) <= 3:
                roe_val = roe
            else:
                roe_val = roe / 100.0
            roe_pct = round(roe_val * 100.0, 2)
        except Exception:
            roe_pct = None

    de = info.get("debtToEquity", np.nan)
    cur_ratio = info.get("currentRatio", np.nan)
    pe = info.get("trailingPE", np.nan)
    peg = info.get("pegRatio", np.nan)
    net_margin = info.get("profitMargins", np.nan)
    beta = info.get("beta", np.nan)
    market_cap = info.get("marketCap", np.nan)

    underv = None
    try:
        if fair_value and current_price and fair_value > 0:
            underv = round(((fair_value - current_price) / fair_value) * 100, 2)
    except Exception:
        underv = None

    # Fundamentals (20)
    if isinstance(de, (int, float)):
        if de < 0.5:
            score += 10; reasons.append("Excellent D/E (<0.5)")
        elif de < 1:
            score += 5; reasons.append("Moderate D/E (<1)")

    if isinstance(cur_ratio, (int, float)):
        if cur_ratio > 1.5:
            score += 10; reasons.append("Healthy Current Ratio (>1.5)")
        elif cur_ratio > 1:
            score += 5; reasons.append("Moderate Liquidity")

    # Profitability (20)
    if isinstance(roe_pct, (int, float)):
        if roe_pct > 18:
            score += 10; reasons.append("Strong ROE (>18%)")
        elif roe_pct > 12:
            score += 5; reasons.append("Good ROE (12–18%)")

    if isinstance(net_margin, (int, float)):
        try:
            net_margin_pct = net_margin * 100
            if net_margin_pct > 15:
                score += 10; reasons.append("High Profit Margin (>15%)")
            elif net_margin_pct > 8:
                score += 5; reasons.append("Moderate Profit Margin")
        except Exception:
            pass

    # Growth (20)
    if isinstance(revenue_cagr, (int, float)):
        if revenue_cagr > 10:
            score += 10; reasons.append("Strong Sales Growth (CAGR >10%)")
        elif revenue_cagr > 5:
            score += 5; reasons.append("Moderate Sales Growth (CAGR)")
    else:
        sales_growth = info.get("revenueGrowth", np.nan)
        if isinstance(sales_growth, (int, float)) and sales_growth > 0.10:
            score += 10; reasons.append("Strong Sales Growth (single-year)")
        elif isinstance(sales_growth, (int, float)) and sales_growth > 0.05:
            score += 5; reasons.append("Moderate Sales Growth (single-year)")

    if isinstance(profit_cagr, (int, float)):
        if profit_cagr > 10:
            score += 10; reasons.append("Strong Profit Growth (CAGR >10%)")
        elif profit_cagr > 5:
            score += 5; reasons.append("Moderate Profit Growth (CAGR)")
    else:
        eps_growth = info.get("earningsQuarterlyGrowth", np.nan)
        if isinstance(eps_growth, (int, float)) and eps_growth > 0.10:
            score += 10; reasons.append("Strong EPS Growth (single-year)")
        elif isinstance(eps_growth, (int, float)) and eps_growth > 0.05:
            score += 5; reasons.append("Moderate EPS Growth (single-year)")

    # Valuation (15)
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 20:
            score += 10; reasons.append("Attractive P/E (<20)")
        elif pe < 30:
            score += 5; reasons.append("Fair P/E (<30)")

    if isinstance(peg, (int, float)) and peg < 1.5:
        score += 5; reasons.append("Reasonable PEG (<1.5)")

    # Momentum (15)
    if isinstance(underv, (int, float)):
        if underv >= 25:
            score += 10; reasons.append("Deep undervaluation (>25%)")
        elif underv >= 10:
            score += 5; reasons.append("Undervalued (>10%)")

    # Safety (10)
    if isinstance(beta, (int, float)):
        if beta < 1:
            score += 10; reasons.append("Low Volatility (β<1)")
        elif beta < 1.2:
            score += 5; reasons.append("Moderate Volatility")

    final_score = min(score, 100)
    rec = "Hold"
    if final_score >= 85:
        rec = "Strong Buy"
    elif final_score >= 70:
        rec = "Buy"
    elif final_score < 55:
        rec = "Avoid"

    return {
        "score": final_score,
        "reasons": reasons,
        "undervaluation_%": underv,
        "recommendation": rec,
        "market_cap": market_cap
    }

# -------------------------
# RJ Score: same function as before (unchanged)
# -------------------------
def stock_score(
    roe,
    debt_eq,
    rev_cagr,
    prof_cagr,
    pe_ratio,
    pe_industry,
    div_yield,
    promoter_hold,
    management_quality=3,
    moat_strength=3,
    growth_potential=3,
    market_phase="neutral"
):
    score = 0
    if roe > 15:
        score += 15
    if debt_eq < 1:
        score += 15
    if rev_cagr > 10:
        score += 10
    if prof_cagr > 10:
        score += 10
    if pe_ratio < pe_industry:
        score += 10
    if div_yield > 1:
        score += 5
    if promoter_hold > 50:
        score += 10
    qualitative = ((management_quality * 4) + (moat_strength * 3) + (growth_potential * 3))
    score += qualitative * 0.6
    if market_phase == "bull":
        score += 5
    elif market_phase == "bear":
        score -= 5
    score = max(0, min(100, round(score, 1)))
    if score >= 90:
        rating = "💎 Strong Buy"
    elif score >= 75:
        rating = "✅ Buy"
    elif score >= 60:
        rating = "🟨 Hold"
    else:
        rating = "🔴 Avoid"
    return {"Score": score, "Rating": rating}

# -------------------------
# Email sender (unchanged)
# -------------------------
def send_email_smtp(smtp_host, smtp_port, username, password, sender, recipients, subject, body):
    try:
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(",") if r.strip()]
        msg = EmailMessage()
        msg["From"] = sender or username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        if smtp_port == 587:
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as ex:
        return False, str(ex)

# -------------------------
# UI Tabs (Dashboard, Single, Portfolio, Alerts, Watchlist, RJ Score)
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor", "🏆 RJ Score"]
)

# -------------------------
# Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard (Alpha Vantage)")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    elif st.button("🔍 Analyze Watchlist"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            sym_av = normalize_for_av(sym)
            # fetch overview & price history (cached)
            info = build_info_from_overview(sym_av)
            df_hist = fetch_time_series_daily(sym_av, outputsize="compact")
            # build currentPrice and 52W high/low from df_hist
            ltp = None
            if not df_hist.empty:
                ltp = float(df_hist["close"].iloc[-1])
                # 52w window approx 365 days
                last_year = df_hist.index.max() - pd.DateOffset(days=365)
                df_52 = df_hist[df_hist.index >= last_year]
                f52h = float(df_52["high"].max()) if not df_52.empty else float(df_hist["high"].max())
                f52l = float(df_52["low"].min()) if not df_52.empty else float(df_hist["low"].min())
            else:
                f52h = f52l = None
            info["currentPrice"] = ltp
            info["fiftyTwoWeekHigh"] = f52h
            info["fiftyTwoWeekLow"] = f52l
            # financial metrics (cached)
            fin_metrics = get_financial_metrics_av(sym)
            revenue_cagr = fin_metrics.get("revenue_cagr_pct")
            profit_cagr = fin_metrics.get("profit_cagr_pct")
            rec = rule_based_recommendation_av(info, None, ltp, revenue_cagr, profit_cagr)
            # fair value
            fv, method = estimate_fair_value_av(info, sym_av)
            buy, sell = compute_buy_sell(fv)
            cap = rec["market_cap"]
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank_score = (rec["score"] * 2) + (rec["undervaluation_%"] or 0) / 10 + cap_weight
            rows.append({
                "Symbol": sym_av,
                "LTP": ltp,
                "Fair Value": fv,
                "Underv%": rec["undervaluation_%"],
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec["recommendation"],
                "Score": rec["score"],
                "RankScore": round(rank_score, 2),
                "Reasons": "; ".join(rec["reasons"]) if rec.get("reasons") else ""
            })
            progress.progress(int(((i + 1) / len(watchlist)) * 100))
            time.sleep(MOCK_SLEEP)
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="RankScore", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Deep Analysis (Alpha Vantage)")
    ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE, RELIANCE.BSE, TCS.NS):")
    if ticker:
        base = ticker.strip().upper()
        sym_av = normalize_for_av(base)
        # fetch overview & history
        info = build_info_from_overview(sym_av)
        df_hist = fetch_time_series_daily(sym_av, outputsize="full")
        # current price + 52w from history
        ltp = None
        if not df_hist.empty:
            ltp = float(df_hist["close"].iloc[-1])
            last_year = df_hist.index.max() - pd.DateOffset(days=365)
            df_52 = df_hist[df_hist.index >= last_year]
            f52h = float(df_52["high"].max()) if not df_52.empty else float(df_hist["high"].max())
            f52l = float(df_52["low"].min()) if not df_52.empty else float(df_hist["low"].min())
        else:
            f52h = f52l = None
        info["currentPrice"] = ltp
        info["fiftyTwoWeekHigh"] = f52h
        info["fiftyTwoWeekLow"] = f52l

        st.subheader("📊 Overview Panel")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{(ltp or 0):,.2f}")
            st.metric("52W High", f"₹{(f52h or 0):,.2f}")
        with col2:
            st.metric("52W Low", f"₹{(f52l or 0):,.2f}")
            mcap = info.get("marketCap", 0) or 0
            st.metric("Market Cap", f"₹{mcap/1e7:,.2f} Cr")
        with col3:
            st.metric("P/E", info.get("trailingPE", "-"))
            st.metric("P/B", info.get("bookValue", "-"))
        with col4:
            dy = info.get("dividendYield", 0) or 0
            st.metric("Dividend Yield", f"{(dy*100 if dy is not None else 0):.2f}%")
            ph = info.get("heldPercentInsiders", 0) or 0
            st.metric("Promoter Holding", f"{(ph*100 if ph is not None else 0):.2f}%")

        st.markdown("---")
        st.subheader("📈 Financial Strength")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roe_display = info.get("returnOnEquity", 0) or 0
            try:
                roe_display_pct = (roe_display * 100) if abs(roe_display) <= 3 else roe_display
            except Exception:
                roe_display_pct = 0
            st.metric("ROE", f"{roe_display_pct:.2f}%")
        with col2:
            # ROA/ROCE not available from overview; show placeholder
            st.metric("ROCE/ROA", "-")
        with col3:
            st.metric("Debt/Equity", f"{(info.get('debtToEquity') or 0):.2f}")
        with col4:
            st.metric("Beta", info.get("beta", "-"))

        # price-based CAGRs (3y & 5y)
        cagr3 = price_cagr_from_df(df_hist, years=3) if not df_hist.empty else None
        cagr5 = price_cagr_from_df(df_hist, years=5) if not df_hist.empty else None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price CAGR (3Y)", f"{'-' if cagr3 is None else str(cagr3) + '%'}")
        with col2:
            st.metric("Price CAGR (5Y)", f"{'-' if cagr5 is None else str(cagr5) + '%'}")
        with col3:
            # EPS change cannot be computed reliably from overview; placeholder
            st.metric("EPS change (est.)", "-")

        st.markdown("---")
        st.subheader("💵 Profitability")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Operating Margin", "-")
        with col2:
            st.metric("Net Profit Margin", "-")
        with col3:
            fcf_info = "-"  # not available in overview
            st.metric("FCF Trend", fcf_info)

        st.markdown("---")
        st.subheader("📉 Valuation Snapshot")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P/E vs Industry", info.get("trailingPE", "-"))
        with col2:
            st.metric("PEG", info.get("pegRatio", "-"))
        with col3:
            dyv = info.get("dividendYield", 0) or 0
            st.metric("Dividend Yield", f"{(dyv*100 if dyv else 0):.2f}%")

        st.markdown("---")
        st.subheader("🧠 RJ STYLE INTERPRETATION")
        st.markdown("""
        > Think like RJ — prefer consistent growth, strong ROE (>15%), low debt, and cash-gen businesses.
        """)

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV (columns: symbol, buy_price, quantity). Symbols without suffix (e.g., RELIANCE).")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf_columns = [c.lower() for c in pf.columns]
            if not set(["symbol", "buy_price", "quantity"]).issubset(set(pf_columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity (case-insensitive)")
            else:
                pf.columns = pf_columns
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    sym_av = normalize_for_av(sym)
                    df = fetch_time_series_daily(sym_av, outputsize="compact")
                    ltp = None
                    if not df.empty:
                        ltp = float(df["close"].iloc[-1])
                    buy = float(r["buy_price"]) if r["buy_price"] not in (None, '') else 0
                    qty = float(r["quantity"]) if r["quantity"] not in (None, '') else 0
                    current_value = round((ltp * qty), 2) if isinstance(ltp, (int, float)) and not math.isnan(ltp) else None
                    invested = round(buy * qty, 2)
                    pl = round((current_value - invested), 2) if current_value is not None else None
                    pl_pct = round((pl / invested * 100), 2) if pl is not None and invested != 0 else None
                    rows.append({
                        "symbol": sym_av,
                        "buy_price": buy,
                        "quantity": qty,
                        "ltp": ltp,
                        "current_value": current_value,
                        "invested": invested,
                        "P/L": pl,
                        "P/L%": pl_pct
                    })
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error reading portfolio: " + str(e))

# -------------------------
# Alerts (Email)
# -------------------------
with tab4:
    st.header("📣 Email Alerts (manual send)")
    st.write("This sends immediate email(s). For Gmail, use smtp.gmail.com port 587 and an App Password.")
    with st.form("alert_form"):
        smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", value=587)
        smtp_user = st.text_input("SMTP username (email)")
        smtp_pass = st.text_input("SMTP password (app password recommended)", type="password")
        sender = st.text_input("From (optional)", value=smtp_user)
        recipients = st.text_input("Recipients (comma separated)")
        underv_threshold = st.number_input("Send alerts when undervaluation% >= ", value=10)
        submit_alert = st.form_submit_button("Send Alerts Now")

    if submit_alert:
        if not smtp_user or not smtp_pass or not recipients:
            st.error("Provide SMTP username/password and recipient(s).")
        else:
            results = []
            wl = load_watchlist()
            for sym in wl:
                sym_av = normalize_for_av(sym)
                info = build_info_from_overview(sym_av)
                df_hist = fetch_time_series_daily(sym_av, outputsize="compact")
                ltp = float(df_hist["close"].iloc[-1]) if not df_hist.empty else None
                info["currentPrice"] = ltp
                fv, method = estimate_fair_value_av(info, sym_av)
                underv = None
                if fv and ltp and fv > 0:
                    underv = round(((fv - ltp) / fv) * 100, 2)
                if isinstance(underv, (int, float)) and underv >= underv_threshold:
                    results.append(f"{sym_av}: LTP ₹{ltp} | Fair ₹{fv} ({method}) | Underval {underv}%")
                time.sleep(MOCK_SLEEP)
            if not results:
                st.info("No stocks passed the threshold.")
            else:
                body = "StockMentor alerts:\n\n" + "\n".join(results) + f"\n\nGenerated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ok, msg = send_email_smtp(smtp_host, int(smtp_port), smtp_user, smtp_pass, sender, recipients, "StockMentor Alerts", body)
                if ok:
                    st.success("Alerts sent successfully.")
                else:
                    st.error("Failed to send alerts: " + msg)

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    st.write("Edit your watchlist (one symbol per line). Use base tickers (without suffix) or include .BSE/.NS if you prefer.")
    current = load_watchlist()
    new_txt = st.text_area("Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("Watchlist saved. Reload Dashboard to analyze.")
        else:
            st.error("Save failed: " + msg)

# -------------------------
# RJ Score Tab
# -------------------------
with tab6:
    st.header("🏆 RJ Score — Jhunjhunwala-Style Hybrid Stock Scoring System (Alpha Vantage)")
    st.markdown("""
    Combines fundamental proxies (overview) + price-based growth (CAGR) when fundamentals missing.
    """)

    watchlist = load_watchlist()
    if not watchlist:
        st.info("⚠️ Watchlist empty. Add symbols in Watchlist Editor.")
    else:
        with st.expander("Scoring parameters / defaults"):
            market_phase = st.selectbox("Market Phase", ["neutral", "bull", "bear"], index=0)
            st.write("Default subjective ratings used for all stocks below. You can change them and re-run scoring.")
            management_quality = st.slider("Management quality (1-5)", 1, 5, 4)
            moat_strength = st.slider("Moat strength (1-5)", 1, 5, 3)
            growth_potential = st.slider("Growth potential (1-5)", 1, 5, 4)

        if st.button("🏁 Run RJ Scoring"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                sym_av = normalize_for_av(sym)
                info = build_info_from_overview(sym_av)
                df_hist = fetch_time_series_daily(sym_av, outputsize="compact")
                ltp = float(df_hist["close"].iloc[-1]) if not df_hist.empty else None
                info["currentPrice"] = ltp

                fin_metrics = get_financial_metrics_av(sym)
                roe_display = fin_metrics.get("roe_pct") or 0
                debt_eq = fin_metrics.get("debt_to_equity") or 0
                rev_cagr = fin_metrics.get("revenue_cagr_pct") or 0
                prof_cagr = fin_metrics.get("profit_cagr_pct") or 0
                pe_ratio = info.get("trailingPE") or DEFAULT_PE_TARGET
                pe_industry = info.get("forwardPE") or DEFAULT_PE_TARGET
                div_yield = fin_metrics.get("dividend_yield_pct") or 0
                promoter_hold = fin_metrics.get("promoter_holding_pct") or 0

                # sanity
                if rev_cagr is None or math.isnan(rev_cagr):
                    rev_cagr = 0
                if prof_cagr is None or math.isnan(prof_cagr):
                    prof_cagr = 0

                if rev_cagr > 100 or rev_cagr < -50:
                    rev_cagr = 0
                if prof_cagr > 100 or prof_cagr < -50:
                    prof_cagr = 0

                result = stock_score(
                    roe_display or 0,
                    debt_eq or 0,
                    rev_cagr or 0,
                    prof_cagr or 0,
                    pe_ratio or DEFAULT_PE_TARGET,
                    pe_industry or DEFAULT_PE_TARGET,
                    div_yield or 0,
                    promoter_hold or 0,
                    management_quality,
                    moat_strength,
                    growth_potential,
                    market_phase
                )

                rows.append({
                    "Symbol": sym_av,
                    "ROE%": roe_display,
                    "D/E": round(debt_eq or 0, 2),
                    "Rev CAGR%": round(rev_cagr, 1),
                    "Profit CAGR%": round(prof_cagr, 1),
                    "Div Yield%": round(div_yield, 2),
                    "Promoter%": round(promoter_hold, 1),
                    "RJ Score": result["Score"],
                    "Rating": result["Rating"]
                })
                progress.progress(int(((i + 1) / len(watchlist)) * 100))
                time.sleep(MOCK_SLEEP)
            df = pd.DataFrame(rows)
            df_sorted = df.sort_values(by="RJ Score", ascending=False)
            st.dataframe(df_sorted, use_container_width=True)
            st.success("✅ RJ-style ranking complete — blending fundamentals with conviction!")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("StockMentor (Alpha Vantage) — Data via Alpha Vantage. Keep API key secure and watch free-tier rate limits.")
