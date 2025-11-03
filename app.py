# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor
- Rule-based scoring, ranking & recommendation
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
        load_watchlist.clear()
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
    if not fair_value or math.isnan(fair_value):
        return None, None
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos/1.5), 2)

# -------------------------
# Rule-based recommendation
# -------------------------
def rule_based_recommendation(info, fair_value, current_price):
    """
    100-point rule-based long-term scoring:
    Fundamentals, Profitability, Growth, Valuation, Momentum, Safety.
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
    eps_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    sales_growth = safe_get(info, "revenueGrowth", np.nan)
    beta = safe_get(info, "beta", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)

    underv = None
    if fair_value and current_price and fair_value > 0:
        underv = round(((fair_value - current_price) / fair_value) * 100, 2)

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
    if isinstance(sales_growth, (int, float)):
        if sales_growth > 0.10:
            score += 10; reasons.append("Strong Sales Growth (>10%)")
        elif sales_growth > 0.05:
            score += 5; reasons.append("Moderate Sales Growth")
    if isinstance(eps_growth, (int, float)):
        if eps_growth > 0.10:
            score += 10; reasons.append("Strong EPS Growth (>10%)")
        elif eps_growth > 0.05:
            score += 5; reasons.append("Moderate EPS Growth")

    # --- 4. Valuation (15 pts) ---
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 20:
            score += 10; reasons.append("Attractive P/E (<20)")
        elif pe < 30:
            score += 5; reasons.append("Fair P/E (<30)")
    if isinstance(peg, (int, float)) and peg < 1.5:
        score += 5; reasons.append("Reasonable PEG (<1.5)")

    # --- 5. Momentum (15 pts) ---
    # (Simplified: check price vs fair value)
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor"])

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
            rec = rule_based_recommendation(info, fv, ltp)
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
                "Reasons": "; ".join(rec["reasons"])
            })
            progress.progress(int(((i+1)/len(watchlist))*100))
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="RankScore", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# Single Stock, Portfolio, Alerts, Watchlist (same as before)
# -------------------------
# (You can retain the same code blocks from your previous version here unchanged)
# -------------------------
# TAB: Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Detail")
    watchlist = load_watchlist()
    sel = st.selectbox("Select stock", watchlist) if watchlist else st.text_input("Enter symbol (e.g., RELIANCE)")

    if sel:
        info, hist = fetch_info_and_history(sel)
        if info.get("error"):
            st.error("Data fetch error: " + info.get("error"))
        else:
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)

            c1, c2, c3 = st.columns(3)
            c1.metric("LTP", f"₹{round(ltp,2) if isinstance(ltp,(int,float)) and not math.isnan(ltp) else '-'}")
            c2.metric("Fair Value", f"₹{fv}" if fv else "-")
            c3.metric("Recommendation", rec.get("recommendation"))

            st.write("**Quick fundamentals**")
            fund = {
                "PE": safe_get(info, "trailingPE"),
                "EPS (TTM)": safe_get(info, "trailingEps"),
                "ROE": safe_get(info, "returnOnEquity"),
                "Debt/Equity": safe_get(info, "debtToEquity"),
                "Market Cap": safe_get(info, "marketCap"),
            }
            st.json(fund)

            st.write("**Valuation details**")
            st.write(f"- Valuation method: {method}")
            st.write(f"- Buy below: ₹{buy}" if buy else "-")
            st.write(f"- Sell above: ₹{sell}" if sell else "-")
            st.write(f"- Undervaluation %: {rec.get('undervaluation_%')}")
            st.write("**Rule-based reasons**")
            st.write(", ".join(rec.get("reasons") or []))

            st.write("**5-year price chart**")
            if hist is not None and not hist.empty:
                st.line_chart(hist["Close"])
            else:
                st.info("No historical price data available.")

# -------------------------
# TAB: Portfolio
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
# TAB: Alerts (Email)
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
# TAB: Watchlist Editor
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
# Footer
# -------------------------
st.markdown("---")
st.caption("StockMentor — rule-based long-term stock helper. Data via Yahoo Finance (yfinance).")

