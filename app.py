# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- Rule-based scoring, ranking & recommendation
Author: Biswanath Das
Updated: corrected CAGR logic, robust financial parsing, RJ scoring fixes
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.message import EmailMessage
from datetime import datetime
import math
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="StockMentor (Rule-based)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("No OpenAI. Pure rule-based valuation, ranking, and recommendations.")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
DISCOUNT_RATE = 0.10
MOCK_SLEEP = 0.02

# -------------------------
# Utilities
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
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=900)
def fetch_info_and_history(symbol_no_suffix):
    symbol = f"{symbol_no_suffix}.NS"
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="5y", interval="1d")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()


def safe_get(info, key, default=np.nan):
    v = info.get(key, default) if isinstance(info, dict) else default
    return default if v in (None, "None", "") else v


# -------------------------
# Robust CAGR helper
# -------------------------

def safe_cagr_from_series(series):
    """Compute realistic CAGR from a pandas Series (oldest → newest)."""
    try:
        series = series.dropna().astype(float)
        if len(series) < 3:  # Need at least 3 years (2 intervals)
            return None
        series = series.sort_index()
        start, end = series.iloc[0], series.iloc[-1]
        if start <= 0 or end <= 0:
            return None

        # Ignore cases where base year is too tiny vs. final year (IPO/new company)
        if end / start > 100 and start < 1e7:
            return None

        n = len(series) - 1
        cagr = ((end / start) ** (1 / n) - 1) * 100

        # Cap unrealistic growth
        if cagr > 100 or cagr < -50:
            return None

        return round(cagr, 2)
    except Exception:
        return None




# -------------------------
# Financial metrics helper (robust)
# -------------------------
@st.cache_data(ttl=900)
def get_financial_metrics(symbol_no_suffix):
    """
    Fetches annual financials via ticker.financials and computes n-year CAGRs for Revenue and Profit.
    Returns a dict with safe values or None where unavailable.
    """
    symbol = f"{symbol_no_suffix}.NS"
    out = {
        'revenue_cagr_pct': None,
        'profit_cagr_pct': None,
        'roe_pct': None,
        'debt_to_equity': None,
        'dividend_yield_pct': None,
        'promoter_holding_pct': None,
    }
    try:
        ticker = yf.Ticker(symbol)
        fin = ticker.financials  # columns are years, rows are items
        info = ticker.info or {}

        if fin is not None and not fin.empty:
            # Transpose to get years as index (columns originally are different years)
            fin_t = fin.T.copy()
            # attempt to sort chronologically if index parseable
            try:
                fin_t.index = pd.to_datetime(fin_t.index)
                fin_t = fin_t.sort_index()
            except Exception:
                # if cannot parse index, preserve as-is
                pass

            # fuzzy find revenue / net income columns
            def find_col_like(cols, patterns):
                cols_list = list(cols)
                for p in patterns:
                    for c in cols_list:
                        if p.lower() in str(c).lower():
                            return c
                return None

            revenue_key = find_col_like(fin.index, ['total revenue', 'total_revenue', 'revenue', 'revenues', 'totalrevenues'])
            profit_key = find_col_like(fin.index, ['net income', 'netincome', 'net income applicable', 'netincomeavailable', 'netincomeavailableto'])

            # fallback: search for keywords
            if revenue_key is None:
                matches = [r for r in fin.index if 'reven' in str(r).lower()]
                revenue_key = matches[0] if matches else None
            if profit_key is None:
                matches = [r for r in fin.index if 'net' in str(r).lower() and 'income' in str(r).lower()]
                profit_key = matches[0] if matches else None

            # compute cagr using safe helper
            if revenue_key is not None:
                try:
                    rev_series = fin_t[revenue_key]
                    rev_cagr = safe_cagr_from_series(rev_series)
                    out['revenue_cagr_pct'] = rev_cagr
                except Exception:
                    out['revenue_cagr_pct'] = None

            if profit_key is not None:
                try:
                    prof_series = fin_t[profit_key]
                    profit_cagr = safe_cagr_from_series(prof_series)
                    out['profit_cagr_pct'] = profit_cagr
                except Exception:
                    out['profit_cagr_pct'] = None

        # fallback single-value retrieval from info
        roe = safe_get(info, 'returnOnEquity')
        if isinstance(roe, (int, float)):
            # sometimes already in percent (rare) or in fraction
            if abs(roe) <= 3:
                out['roe_pct'] = round(roe * 100, 2)
            else:
                out['roe_pct'] = round(roe, 2)

        debt_eq = safe_get(info, 'debtToEquity', np.nan)
        out['debt_to_equity'] = None if pd.isna(debt_eq) else debt_eq

        div_yield = safe_get(info, 'dividendYield', np.nan)
        if isinstance(div_yield, (int, float)):
            out['dividend_yield_pct'] = round(div_yield * 100, 2)

        promoter_hold = safe_get(info, 'heldPercentInsiders', np.nan)
        if isinstance(promoter_hold, (int, float)):
            out['promoter_holding_pct'] = round(promoter_hold * 100, 2)

        return out

    except Exception:
        return out


# -------------------------
# Fair Value Estimation
# -------------------------
def estimate_fair_value(info):
    try:
        target = safe_get(info, "targetMeanPrice", np.nan)
        if isinstance(target, (int, float)) and target > 0:
            return round(float(target), 2), "AnalystTarget"
    except Exception:
        pass

    eps = safe_get(info, "trailingEps", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    trailing_pe = safe_get(info, "trailingPE", np.nan)

    pe_target = DEFAULT_PE_TARGET
    if isinstance(forward_pe, (int, float)) and forward_pe > 0 and forward_pe < 200:
        pe_target = forward_pe
    elif isinstance(trailing_pe, (int, float)) and trailing_pe > 0 and trailing_pe < 200:
        pe_target = max(10.0, trailing_pe * 0.9)

    if isinstance(eps, (int, float)) and eps > 0:
        fv = eps * pe_target
        return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"

    book = safe_get(info, "bookValue", np.nan)
    if isinstance(book, (int, float)) and book > 0 and isinstance(trailing_pe, (int, float)) and trailing_pe > 0:
        fv = book * trailing_pe
        return round(float(fv), 2), "BVxPE"

    return None, "InsufficientData"


# -------------------------
# Buy/Sell price zones
# -------------------------
def compute_buy_sell(fair_value, mos=0.30):
    if fair_value is None or (isinstance(fair_value, float) and math.isnan(fair_value)):
        return None, None
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos / 1.5), 2)


# -------------------------
# Rule-based recommendation (updated to use robust CAGRs)
# -------------------------
def rule_based_recommendation(info, fair_value, current_price, revenue_cagr=None, profit_cagr=None):
    score = 0
    reasons = []

    roe = safe_get(info, "returnOnEquity", np.nan)
    if isinstance(roe, (int, float)):
        if abs(roe) <= 3:
            roe_val = roe
        else:
            roe_val = roe / 100.0
        roe_pct = None
        try:
            roe_pct = round((roe_val * 100), 2)
        except Exception:
            roe_pct = None
    else:
        roe_pct = None

    de = safe_get(info, "debtToEquity", np.nan)
    cur_ratio = safe_get(info, "currentRatio", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)
    peg = safe_get(info, "pegRatio", np.nan)
    net_margin = safe_get(info, "profitMargins", np.nan)
    beta = safe_get(info, "beta", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)

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

    # Growth (20) - uses robust CAGRs
    if isinstance(revenue_cagr, (int, float)):
        if revenue_cagr > 10:
            score += 10; reasons.append("Strong Sales Growth (CAGR >10%)")
        elif revenue_cagr > 5:
            score += 5; reasons.append("Moderate Sales Growth (CAGR)")
    else:
        sales_growth = safe_get(info, "revenueGrowth", np.nan)
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
        eps_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
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
# RJ Score: Jhunjhunwala-Style Hybrid Scoring
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

    # 1️⃣ Fundamental Strength (max ~75)
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

    # 2️⃣ Qualitative Conviction (scaled 0–30)
    qualitative = (
        (management_quality * 4) + (moat_strength * 3) + (growth_potential * 3)
    )  # max 50 → scaled to 30
    score += qualitative * 0.6  # 30 max

    # 3️⃣ Market Cycle Adjustment
    if market_phase == "bull":
        score += 5
    elif market_phase == "bear":
        score -= 5

    # 4️⃣ Cap and label
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
# Email sender
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
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor", "🏆 RJ Score"]
)

# -------------------------
# Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    elif st.button("🔍 Analyze Watchlist"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            info, _ = fetch_info_and_history(sym)
            if info.get("error"):
                continue
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, method = estimate_fair_value(info)
            fin_metrics = get_financial_metrics(sym)
            rec = rule_based_recommendation(info, fv, ltp, fin_metrics.get('revenue_cagr_pct'), fin_metrics.get('profit_cagr_pct'))
            buy, sell = compute_buy_sell(fv)
            cap = rec["market_cap"]
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank_score = (rec["score"] * 2) + (rec["undervaluation_%"] or 0) / 10 + cap_weight
            rows.append({
                "Symbol": sym,
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
# Single Stock - corrected
# -------------------------
with tab2:
    st.header("📈 Single Stock Deep Analysis (RJ Style)")

    ticker = st.text_input("Enter Stock Symbol (e.g., TCS.NS, HDFCBANK.NS, INFY.NS):")

    if ticker:
        # Normalize symbol
        symbol = ticker.strip().upper().replace('.NS', '')
        stock = yf.Ticker(f"{symbol}.NS")
        info = stock.info

        st.subheader("📊 Overview Panel")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{safe_get(info, 'currentPrice', 0):,.2f}")
            st.metric("52W High", f"₹{safe_get(info, 'fiftyTwoWeekHigh', 0):,.2f}")
        with col2:
            st.metric("52W Low", f"₹{safe_get(info, 'fiftyTwoWeekLow', 0):,.2f}")
            mcap = safe_get(info, 'marketCap', 0) or 0
            st.metric("Market Cap", f"₹{mcap/1e7:,.2f} Cr")
        with col3:
            st.metric("P/E", safe_get(info, "trailingPE", "-"))
            st.metric("P/B", safe_get(info, "priceToBook", "-"))
        with col4:
            dy = safe_get(info, 'dividendYield', 0) or 0
            st.metric("Dividend Yield", f"{dy*100:.2f}%")
            ph = safe_get(info, 'heldPercentInsiders', 0) or 0
            st.metric("Promoter Holding", f"{ph*100:.2f}%")

        st.markdown("---")

        st.subheader("📈 Financial Strength")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roe_display = safe_get(info, 'returnOnEquity', 0)
            try:
                roe_display_pct = (roe_display * 100) if abs(roe_display) <= 3 else roe_display
            except Exception:
                roe_display_pct = 0
            st.metric("ROE", f"{roe_display_pct:.2f}%")
        with col2:
            roa = safe_get(info, 'returnOnAssets', 0)
            try:
                roa_pct = (roa * 100) if abs(roa) <= 3 else roa
            except Exception:
                roa_pct = 0
            st.metric("ROCE/ROA", f"{roa_pct:.2f}%")
        with col3:
            st.metric("Debt/Equity", f"{safe_get(info, 'debtToEquity', 0):.2f}")
        with col4:
            st.metric("Interest Coverage", safe_get(info, 'interestCoverage', "-"))

        # Robust CAGR computation for Single Stock
        try:
            fin = stock.financials
            if fin is not None and not fin.empty:
                fin_t = fin.T.copy()
                try:
                    fin_t.index = pd.to_datetime(fin_t.index)
                    fin_t = fin_t.sort_index()
                except Exception:
                    pass

                # fuzzy find revenue and net income
                def find_item_like(df_index, patterns):
                    for p in patterns:
                        for itm in df_index:
                            if p.lower() in str(itm).lower():
                                return itm
                    return None

                rev_item = find_item_like(fin.index, ['total revenue', 'revenue', 'total_revenue', 'revenues'])
                net_item = find_item_like(fin.index, ['net income', 'netincome', 'net income applicable', 'netincomeavailable'])

                rev_cagr = None
                profit_cagr = None
                rev_periods = None

                if rev_item is not None:
                    rev_cagr = safe_cagr_from_series(fin_t[rev_item])
                if net_item is not None:
                    profit_cagr = safe_cagr_from_series(fin_t[net_item])

            else:
                rev_cagr = profit_cagr = None
        except Exception:
            rev_cagr = profit_cagr = None

        eps_cagr = None
        try:
            # simple forward/trailing eps change as a fallback (not CAGR)
            trailing_eps = safe_get(info, 'trailingEps', None)
            forward_eps = safe_get(info, 'forwardEps', None)
            if isinstance(trailing_eps, (int, float)) and isinstance(forward_eps, (int, float)) and trailing_eps != 0:
                eps_cagr = round(((forward_eps / trailing_eps) - 1) * 100, 2)
        except Exception:
            eps_cagr = None

        col1, col2, col3 = st.columns(3)
        with col1:
            label = f"Revenue CAGR ({'nY' if rev_cagr is not None else 'nY'})"
            st.metric("Revenue CAGR (nY)", f"{'-' if rev_cagr is None else str(rev_cagr) + '%'}")
        with col2:
            st.metric("Profit CAGR (nY)", f"{'-' if profit_cagr is None else str(profit_cagr) + '%'}")
        with col3:
            st.metric("EPS change (forward vs trailing)", f"{'-' if eps_cagr is None else str(eps_cagr) + '%'}")

        st.markdown("---")

        st.subheader("💵 Profitability")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Operating Margin", f"{safe_get(info, 'operatingMargins', 0)*100:.2f}%")
        with col2:
            st.metric("Net Profit Margin", f"{safe_get(info, 'profitMargins', 0)*100:.2f}%")
        with col3:
            fcf = safe_get(info, 'freeCashflow', 0) or 0
            st.metric("FCF Trend", "↑ Positive" if fcf > 0 else "↓ Negative")

        try:
            fin_display = None
            if fin is not None and not fin.empty:
                fin_disp = fin.T.copy()
                # try to pick common columns if present
                cols = []
                for c in ['Total Revenue', 'Gross Profit', 'Net Income', 'Revenue', 'Net Income Available']:
                    if c in fin_disp.columns:
                        cols.append(c)
                # fallback to any revenue/net columns
                if len(cols) < 2:
                    cols = fin_disp.columns[:3].tolist()
                fin_small = fin_disp[cols] / 1e7
                fin_small.columns = [str(c) + ' (Cr)' for c in fin_small.columns]
                st.line_chart(fin_small)
        except Exception:
            st.warning("Unable to display Profit Trend chart.")

        st.markdown("---")

        st.subheader("📉 Valuation Snapshot")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P/E vs Industry", safe_get(info, "trailingPE", "-"))
        with col2:
            st.metric("EV/EBITDA", safe_get(info, "enterpriseToEbitda", "-"))
        with col3:
            dyv = safe_get(info, 'dividendYield', 0) or 0
            st.metric("Dividend Yield", f"{dyv*100:.2f}%")

        st.markdown("---")

        st.subheader("🧠 RJ STYLE INTERPRETATION")
        st.markdown("""
        > **Think like RJ (Rakesh Jhunjhunwala):**  
        - Look for **consistent growth** in revenue & profits.  
        - **ROE > 15%** and **low Debt/Equity (<0.5)** indicate quality.  
        - Avoid hype; prefer **cash-generating, scalable businesses**.  
        - A great business can **compound earnings** over time with strong management & moat.  
        """)

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV (columns: symbol, buy_price, quantity). Symbols should be without '.NS' (e.g., RELIANCE).")
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
                    buy = float(r["buy_price"]) if r["buy_price"] not in (None, '') else 0
                    qty = float(r["quantity"]) if r["quantity"] not in (None, '') else 0
                    info, _ = fetch_info_and_history(sym)
                    ltp = safe_get(info, "currentPrice", np.nan)
                    current_value = round((ltp * qty), 2) if isinstance(ltp, (int, float)) and not math.isnan(ltp) else None
                    invested = round(buy * qty, 2)
                    pl = round((current_value - invested), 2) if current_value is not None else None
                    pl_pct = round((pl / invested * 100), 2) if pl is not None and invested != 0 else None
                    rows.append({
                        "symbol": sym,
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
                info, _ = fetch_info_and_history(sym)
                if info.get("error"):
                    continue
                ltp = safe_get(info, "currentPrice", np.nan)
                fv, method = estimate_fair_value(info)
                underv = None
                if fv and ltp and fv > 0:
                    underv = round(((fv - ltp) / fv) * 100, 2)
                if isinstance(underv, (int, float)) and underv >= underv_threshold:
                    results.append(f"{sym}: LTP ₹{ltp} | Fair ₹{fv} ({method}) | Underval {underv}%")
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
    st.write("Edit your watchlist (one symbol per line). Use NSE tickers (without .NS).")
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
    st.header("🏆 RJ Score — Jhunjhunwala-Style Hybrid Stock Scoring System")
    st.markdown("""
    **Author:** Biswanath Das (StockMentor)
    **Inspired by:** Rakesh Jhunjhunwala’s long-term investing philosophy.  
    Combines:  
    1️⃣ *Fundamental Strength* (data-driven)  
    2️⃣ *Qualitative Conviction* (judgment-based)  
    3️⃣ *Market Cycle Adjustment* (macro awareness)
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
                info, _ = fetch_info_and_history(sym)
                if info.get("error"):
                    continue

                # pull 3Y CAGRs using helper
                fin_metrics = get_financial_metrics(sym)

                roe_display = fin_metrics.get("roe_pct") or np.nan
                debt_eq = fin_metrics.get("debt_to_equity") or 0
                rev_cagr = fin_metrics.get("revenue_cagr_3y") or 0
                prof_cagr = fin_metrics.get("profit_cagr_3y") or 0
                pe_ratio = safe_get(info, "trailingPE", DEFAULT_PE_TARGET)
                pe_industry = safe_get(info, "forwardPE", DEFAULT_PE_TARGET) or DEFAULT_PE_TARGET
                div_yield = fin_metrics.get("dividend_yield_pct") or 0
                promoter_hold = fin_metrics.get("promoter_holding_pct") or 0

                # ----------------------------
                # ✅ Sanity check for abnormal CAGRs
                # ----------------------------
                if rev_cagr > 100 or rev_cagr < -50:
                    rev_cagr = 0
                if prof_cagr > 100 or prof_cagr < -50:
                    prof_cagr = 0

                # RJ Scoring
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
                    "Symbol": sym,
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
st.caption("StockMentor — rule-based long-term stock helper. Data via Yahoo Finance (yfinance).")
