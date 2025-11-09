# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- Rule-based scoring, ranking & recommendation
Author: Biswanath Das
Updated: added 3-year CAGR financial metrics helper and integrated into UI
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
# Load/save watchlist
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        symbols = [s.replace(".NS", "").strip().upper() for s in symbols if s and str(s).strip()]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading {WATCHLIST_FILE}: {e}")
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        # try to clear cache so UI picks up new watchlist next run
        try:
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Data fetch
# -------------------------
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
    v = info.get(key, default)
    return default if v in (None, "None", "") else v

# -------------------------
# Financial metrics helper (new)
# -------------------------
@st.cache_data(ttl=900)
def get_financial_metrics(symbol_no_suffix):
    """Fetches annual financials via ticker.financials and computes 3-year CAGRs for Revenue and Profit.
    Returns a dict with:
      - revenue_cagr (3Y %) or None
      - profit_cagr (3Y %) or None
      - roe_pct (ROE in % or None)
      - debt_to_equity (or None)
      - dividend_yield_pct (or None)
      - promoter_holding_pct (or None)

    If there are fewer than 4 annual columns in financials, returns revenue_cagr/profit_cagr as None.
    """
    symbol = f"{symbol_no_suffix}.NS"
    try:
        ticker = yf.Ticker(symbol)
        fin = ticker.financials  # Annual income statement (columns are years, most recent first)
        info = ticker.info or {}

        # helpers to find likely revenue/profit rows (index names vary)
        revenue_keys = [
            'Total Revenue', 'TotalRevenue', 'Revenue', 'Revenues', 'Total revenues'
        ]
        profit_keys = [
            'Net Income', 'NetIncome', 'Net Income Applicable To Common Shares', 'Net income', 'NetIncomeLoss'
        ]

        revenue_val = None
        profit_val = None
        revenue_cagr = None
        profit_cagr = None

        if fin is not None and not fin.empty:
            # columns are typically timestamps (most recent first)
            cols = list(fin.columns)
            if len(cols) >= 4:
                # find revenue row
                for rk in revenue_keys:
                    if rk in fin.index:
                        revenue_row = fin.loc[rk]
                        break
                else:
                    # attempt fuzzy: find first row containing 'reven' (case-insensitive)
                    matches = [r for r in fin.index if 'reven' in str(r).lower()]
                    revenue_row = fin.loc[matches[0]] if matches else None

                for pk in profit_keys:
                    if pk in fin.index:
                        profit_row = fin.loc[pk]
                        break
                else:
                    matches = [r for r in fin.index if 'net' in str(r).lower() and 'income' in str(r).lower()]
                    profit_row = fin.loc[matches[0]] if matches else None

                try:
                    if revenue_row is not None:
                        latest = revenue_row.iloc[0]
                        oldest = revenue_row.iloc[3]
                        if pd.notna(latest) and pd.notna(oldest) and oldest != 0:
                            revenue_cagr = ((float(latest) / float(oldest)) ** (1.0 / 3.0) - 1.0) * 100.0
                            revenue_cagr = round(revenue_cagr, 2)
                except Exception:
                    revenue_cagr = None

                try:
                    if profit_row is not None:
                        latest_p = profit_row.iloc[0]
                        oldest_p = profit_row.iloc[3]
                        if pd.notna(latest_p) and pd.notna(oldest_p) and oldest_p != 0:
                            profit_cagr = ((float(latest_p) / float(oldest_p)) ** (1.0 / 3.0) - 1.0) * 100.0
                            profit_cagr = round(profit_cagr, 2)
                except Exception:
                    profit_cagr = None

        # fallback single value retrieval from info
        roe = safe_get(info, 'returnOnEquity')
        roe_pct = None
        if isinstance(roe, (int, float)):
            if abs(roe) <= 3:
                roe_pct = round(roe * 100, 2)
            else:
                roe_pct = round(roe, 2)

        debt_eq = safe_get(info, 'debtToEquity', np.nan)
        div_yield = safe_get(info, 'dividendYield', 0)
        dividend_yield_pct = None
        if isinstance(div_yield, (int, float)):
            dividend_yield_pct = round(div_yield * 100, 2)

        promoter_hold = safe_get(info, 'heldPercentInsiders', np.nan)
        promoter_holding_pct = None
        if isinstance(promoter_hold, (int, float)):
            promoter_holding_pct = round(promoter_hold * 100, 2)

        return {
            'revenue_cagr_3y': revenue_cagr,
            'profit_cagr_3y': profit_cagr,
            'roe_pct': roe_pct,
            'debt_to_equity': debt_eq if not pd.isna(debt_eq) else None,
            'dividend_yield_pct': dividend_yield_pct,
            'promoter_holding_pct': promoter_holding_pct,
        }

    except Exception as e:
        # don't crash UI — return best-effort
        return {
            'revenue_cagr_3y': None,
            'profit_cagr_3y': None,
            'roe_pct': None,
            'debt_to_equity': None,
            'dividend_yield_pct': None,
            'promoter_holding_pct': None,
        }

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

    if isinstance(forward_pe, (int, float)) and forward_pe > 0 and forward_pe < 200:
        pe_target = forward_pe
    elif isinstance(trailing_pe, (int, float)) and trailing_pe > 0 and trailing_pe < 200:
        pe_target = max(10.0, trailing_pe * 0.9)
    else:
        pe_target = DEFAULT_PE_TARGET

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
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos/1.5), 2)

# -------------------------
# Rule-based recommendation (updated to use 3Y CAGRs)
# -------------------------
def rule_based_recommendation(info, fair_value, current_price, revenue_cagr_3y=None, profit_cagr_3y=None):
    """
    100-point rule-based long-term scoring: Fundamentals, Profitability, Growth, Valuation, Momentum, Safety.
    Growth uses 3-year CAGRs when available.
    """
    score = 0
    reasons = []

    # --- Core data ---
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and abs(roe) > 1:
        roe /= 100.0
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

    # --- 1. Fundamentals (20 pts) ---
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

    # --- 2. Profitability (20 pts) ---
    if isinstance(roe, (int, float)):
        if roe > 0.18:
            score += 10; reasons.append("Strong ROE (>18%)")
        elif roe > 0.12:
            score += 5; reasons.append("Good ROE (12–18%)")

    if isinstance(net_margin, (int, float)):
        if net_margin > 0.15:
            score += 10; reasons.append("High Profit Margin (>15%)")
        elif net_margin > 0.08:
            score += 5; reasons.append("Moderate Profit Margin")

    # --- 3. Growth (20 pts) ---
    # Use 3Y CAGRs when available; fallback to single-year fields if not
    if isinstance(revenue_cagr_3y, (int, float)):
        if revenue_cagr_3y > 10:
            score += 10; reasons.append("Strong Sales Growth (3Y CAGR >10%)")
        elif revenue_cagr_3y > 5:
            score += 5; reasons.append("Moderate Sales Growth (3Y CAGR)")
    else:
        sales_growth = safe_get(info, "revenueGrowth", np.nan)
        if isinstance(sales_growth, (int, float)) and sales_growth > 0.10:
            score += 10; reasons.append("Strong Sales Growth (single-year)")
        elif isinstance(sales_growth, (int, float)) and sales_growth > 0.05:
            score += 5; reasons.append("Moderate Sales Growth (single-year)")

    if isinstance(profit_cagr_3y, (int, float)):
        if profit_cagr_3y > 10:
            score += 10; reasons.append("Strong Profit Growth (3Y CAGR >10%)")
        elif profit_cagr_3y > 5:
            score += 5; reasons.append("Moderate Profit Growth (3Y CAGR)")
    else:
        eps_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
        if isinstance(eps_growth, (int, float)) and eps_growth > 0.10:
            score += 10; reasons.append("Strong EPS Growth (single-year)")
        elif isinstance(eps_growth, (int, float)) and eps_growth > 0.05:
            score += 5; reasons.append("Moderate EPS Growth (single-year)")

    # --- 4. Valuation (15 pts) ---
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 20:
            score += 10; reasons.append("Attractive P/E (<20)")
        elif pe < 30:
            score += 5; reasons.append("Fair P/E (<30)")

    if isinstance(peg, (int, float)) and peg < 1.5:
        score += 5; reasons.append("Reasonable PEG (<1.5)")

    # --- 5. Momentum (15 pts) ---
    if isinstance(underv, (int, float)):
        if underv >= 25:
            score += 10; reasons.append("Deep undervaluation (>25%)")
        elif underv >= 10:
            score += 5; reasons.append("Undervalued (>10%)")

    # --- 6. Safety (10 pts) ---
    if isinstance(beta, (int, float)):
        if beta < 1:
            score += 10; reasons.append("Low Volatility (β<1)")
        elif beta < 1.2:
            score += 5; reasons.append("Moderate Volatility")

    # --- Convert to final score ---
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
    """Jhunjhunwala-Style Hybrid Scoring System"""
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
            # fetch 3y cagr metrics (best-effort)
            fin_metrics = get_financial_metrics(sym)
            rec = rule_based_recommendation(info, fv, ltp, fin_metrics.get('revenue_cagr_3y'), fin_metrics.get('profit_cagr_3y'))
            buy, sell = compute_buy_sell(fv)
            cap = rec["market_cap"]
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank_score = (rec["score"] * 2) + (rec["undervaluation_%"] or 0)/10 + cap_weight
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
            progress.progress(int(((i+1)/len(watchlist))*100))
            time.sleep(MOCK_SLEEP)
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="RankScore", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# Single Stock - RJ Style Deep Analysis (Cleaned & Optimized)
# -------------------------

with tab2:
    st.header("📈 Single Stock Deep Analysis (RJ Style)")

    ticker = st.text_input("Enter Stock Symbol (e.g., TCS.NS, HDFCBANK.NS, INFY.NS):")

    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info

        # =====================================================
        # 📊 OVERVIEW PANEL
        # =====================================================
        st.subheader("📊 Overview Panel")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{info.get('currentPrice', 0):,.2f}")
            st.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}")
        with col2:
            st.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}")
            st.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e7:,.2f} Cr")
        with col3:
            st.metric("P/E", info.get("trailingPE", "-"))
            st.metric("P/B", info.get("priceToBook", "-"))
        with col4:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            st.metric("Promoter Holding", f"{info.get('heldPercentInsiders', 0)*100:.2f}%")

        st.caption("FII/DII trend data not available via yfinance — can be integrated via NSE API later.")
        st.markdown("---")

        # =====================================================
        # 📈 FINANCIAL STRENGTH
        # =====================================================
        st.subheader("📈 Financial Strength")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
        with col2:
            st.metric("ROCE", f"{info.get('returnOnAssets', 0)*100:.2f}%")
        with col3:
            st.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
        with col4:
            st.metric("Interest Coverage", info.get("interestCoverage", "-"))

        # CAGR
        try:
            fin = stock.financials.T.tail(3)
            fin_cr = fin / 1e7
            rev_cagr = ((fin["Total Revenue"].iloc[-1] / fin["Total Revenue"].iloc[0]) ** (1/2) - 1) * 100
            profit_cagr = ((fin["Net Income"].iloc[-1] / fin["Net Income"].iloc[0]) ** (1/2) - 1) * 100
        except Exception:
            rev_cagr = profit_cagr = "-"

        eps_cagr = "-"
        if info.get("trailingEps") and info.get("forwardEps"):
            try:
                eps_cagr = ((info.get("forwardEps") / info.get("trailingEps")) - 1) * 100
            except Exception:
                eps_cagr = "-"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue CAGR (3Y)", f"{'-' if isinstance(rev_cagr, str) else f'{rev_cagr:.2f}%'}")
        with col2:
            st.metric("Profit CAGR (3Y)", f"{'-' if isinstance(profit_cagr, str) else f'{profit_cagr:.2f}%'}")
        with col3:
            st.metric("EPS CAGR (3Y)", f"{'-' if isinstance(eps_cagr, str) else f'{eps_cagr:.2f}%'}")

        st.markdown("---")

        # =====================================================
        # 💵 PROFITABILITY
        # =====================================================
        st.subheader("💵 Profitability")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Operating Margin", f"{info.get('operatingMargins', 0)*100:.2f}%")
        with col2:
            st.metric("Net Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
        with col3:
            st.metric("FCF Trend", "↑ Positive" if info.get('freeCashflow', 0) > 0 else "↓ Negative")

        try:
            fin_display = fin_cr[["Total Revenue", "Gross Profit", "Net Income"]]
            fin_display.columns = ["Revenue (Cr)", "Gross Profit (Cr)", "Net Income (Cr)"]
            st.bar_chart(fin_display)
        except Exception:
            st.warning("Unable to display Profit Trend chart.")

        st.markdown("---")

        # =====================================================
        # 📉 VALUATION SNAPSHOT
        # =====================================================
        st.subheader("📉 Valuation Snapshot")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P/E vs Industry", info.get("trailingPE", "-"))
        with col2:
            st.metric("EV/EBITDA", info.get("enterpriseToEbitda", "-"))
        with col3:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

        st.markdown("---")

        # =====================================================
        # 🥧 PROMOTER HOLDING PIE (Plotly)
        # =====================================================
        st.subheader("🥧 Promoter Holding Breakdown")

        try:
            promoter = info.get("heldPercentInsiders", 0)
            promoter = float(promoter or 0) * 100
            others = max(0, 100 - promoter)

            if promoter <= 0:
                st.info("Promoter holding data not available for this stock.")
            else:
                import plotly.express as px
                pie_df = pd.DataFrame({
                    "Category": ["Promoter", "Others"],
                    "Holding %": [promoter, others]
                })
                fig = px.pie(pie_df, names="Category", values="Holding %", color="Category",
                             color_discrete_map={"Promoter": "#2E86C1", "Others": "#AED6F1"},
                             hole=0.3)
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(height=350, title="Shareholding Pattern")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Pie chart rendering error: {e}")

        st.markdown("---")

        # =====================================================
        # 📊 TREND CHARTS
        # =====================================================
        st.subheader("📊 Trend Charts")

        col1, col2 = st.columns(2)
        with col1:
            try:
                fin5 = stock.financials.T.tail(5) / 1e7
                fin5.columns = [c + " (Cr)" for c in fin5.columns]
                st.caption("5Y Revenue vs Profit (₹ Cr)")
                st.line_chart(fin5[["Total Revenue (Cr)", "Net Income (Cr)"]])
            except Exception:
                st.warning("Unable to fetch 5Y Revenue & Profit data.")
        with col2:
            try:
                roe = info.get('returnOnEquity', 0)*100
                margin = info.get('profitMargins', 0)*100
                trend_df = pd.DataFrame({
                    'Metric': ['ROE', 'Profit Margin'],
                    'Value': [roe, margin]
                }).set_index("Metric")
                st.caption("ROE & Profit Margin Trend")
                st.bar_chart(trend_df)
            except Exception:
                st.warning("Unable to display ROE & Margin trend.")

        st.markdown("---")

        # =====================================================
        # 🧠 RJ STYLE INTERPRETATION
        # =====================================================
        st.subheader("🧠 RJ Style Interpretation")
        st.markdown("""
        > **Think like RJ (Rakesh Jhunjhunwala):**  
        - Look for **consistent growth** in revenue & profits.  
        - **ROE > 15%** and **low Debt/Equity (<0.5)** indicate quality.  
        - Avoid hype; prefer **cash-generating, scalable businesses**.  
        - “**Money is made by sitting, not trading.**”  
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
            if not set(["symbol","buy_price","quantity"]).issubset(set(pf_columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity (case-insensitive)")
            else:
                # normalize
                pf.columns = pf_columns
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    info, _ = fetch_info_and_history(sym)
                    ltp = safe_get(info, "currentPrice", np.nan)
                    current_value = round((ltp * qty), 2) if isinstance(ltp,(int,float)) and not math.isnan(ltp) else None
                    invested = round(buy*qty,2)
                    pl = round((current_value - invested),2) if current_value is not None else None
                    pl_pct = round((pl/invested*100),2) if pl is not None and invested !=0 else None
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
                if fv and ltp and fv>0:
                    underv = round(((fv - ltp)/fv)*100,2)
                if isinstance(underv, (int,float)) and underv >= underv_threshold:
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
with tab5:
    st.header("Watchlist Editor")
    watchlist = load_watchlist()
    new_symbol = st.text_input("Add Symbol (e.g., TCS)").upper()

    if st.button("Add to Watchlist"):
        if new_symbol and new_symbol not in watchlist:
            watchlist.append(new_symbol)
            save_watchlist(watchlist)
            st.success(f"{new_symbol} added!")
        else:
            st.warning("Symbol already in list or invalid.")

    if watchlist:
        st.write("### Current Watchlist")
        st.dataframe(pd.DataFrame({"Symbol": watchlist}))

        remove = st.text_input("Remove Symbol").upper()
        if st.button("Remove"):
            if remove in watchlist:
                watchlist.remove(remove)
                save_watchlist(watchlist)
                st.success(f"{remove} removed!")
            else:
                st.warning("Symbol not found.")


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

                # pull 3Y CAGRs using our helper
                fin_metrics = get_financial_metrics(sym)

                roe_display = fin_metrics.get('roe_pct') or np.nan
                debt_eq = fin_metrics.get('debt_to_equity') or 0
                rev_cagr = fin_metrics.get('revenue_cagr_3y') or 0
                prof_cagr = fin_metrics.get('profit_cagr_3y') or 0
                pe_ratio = safe_get(info, "trailingPE", DEFAULT_PE_TARGET)
                pe_industry = safe_get(info, "forwardPE", DEFAULT_PE_TARGET) or DEFAULT_PE_TARGET
                div_yield = fin_metrics.get('dividend_yield_pct') or 0
                promoter_hold = fin_metrics.get('promoter_holding_pct') or 0

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
