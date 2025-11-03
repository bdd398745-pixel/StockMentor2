# app.py
"""
📊 StockMentor — Rule-based Long-Term Stock Analyst (India)
Enhanced UI Edition (Full 5 Tabs)
Author: Adapted for Biswanath Das
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.message import EmailMessage
from datetime import datetime
import math

# -------------------------
# Page Setup & Styling
# -------------------------
st.set_page_config(page_title="📈 StockMentor (Rule-based)", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    /* Global font and layout */
    body {font-family: 'Segoe UI', sans-serif;}
    .stMetric {border-radius: 10px; padding: 12px; text-align: center;}
    .metric-green {background-color: #e8f5e9;}
    .metric-red {background-color: #ffebee;}
    .metric-blue {background-color: #e3f2fd;}
    div[data-testid="stDataFrame"] {border-radius: 8px; overflow: hidden;}
    .stButton>button {border-radius: 8px; font-weight: 600; width: 100%;}
    hr {border: 1px solid #ddd;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Enhanced Visual Dashboard • Rule-based Logic • Data via Yahoo Finance (yfinance)")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
DISCOUNT_RATE = 0.10

# -------------------------
# Watchlist I/O
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return [s.replace(".NS", "").strip().upper() for s in df[0] if str(s).strip()]
    except:
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        load_watchlist.clear()
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Data Fetch
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol_no_suffix):
    try:
        ticker = yf.Ticker(f"{symbol_no_suffix}.NS")
        return ticker.info or {}, ticker.history(period="5y", interval="1d")
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
        t = safe_get(info, "targetMeanPrice", np.nan)
        if isinstance(t, (int, float)) and t > 0:
            return round(float(t), 2), "AnalystTarget"
    except:
        pass
    eps = safe_get(info, "trailingEps", np.nan)
    pe = safe_get(info, "trailingPE", DEFAULT_PE_TARGET)
    if isinstance(eps, (int, float)) and eps > 0:
        return round(eps * pe, 2), f"EPS×PE({pe})"
    return None, "InsufficientData"

def compute_buy_sell(fv, mos=0.30):
    if not fv or math.isnan(fv):
        return None, None
    return round(fv * (1 - mos), 2), round(fv * (1 + mos / 1.5), 2)

# -------------------------
# Recommendation Logic
# -------------------------
def rule_based_recommendation(info, fv, ltp):
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and abs(roe) > 1:
        roe /= 100
    de = safe_get(info, "debtToEquity", np.nan)
    growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    mc = safe_get(info, "marketCap", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)

    underval = round(((fv - ltp) / fv) * 100, 2) if fv and ltp else None
    score, reasons = 0, []

    # Scoring
    if roe >= 0.20: score += 3; reasons.append("High ROE")
    elif roe >= 0.12: score += 2; reasons.append("Good ROE")
    if de <= 0.5: score += 2; reasons.append("Low D/E")
    elif de <= 1.5: score += 1; reasons.append("Moderate D/E")
    if growth >= 0.20: score += 2; reasons.append("Strong Growth")
    elif growth >= 0.05: score += 1; reasons.append("Moderate Growth")
    if underval >= 25: score += 3; reasons.append("Deep Undervaluation")
    elif underval >= 10: score += 2; reasons.append("Undervalued")

    rec = "Hold"
    if score >= 7 and underval >= 10:
        rec = "Strong Buy"
    elif score >= 5 and underval >= 5:
        rec = "Buy"
    elif pe > 80 or roe < 0:
        rec = "Avoid"

    return {"score": score, "reasons": reasons, "undervaluation_%": underval, "recommendation": rec, "market_cap": mc}

# -------------------------
# Email Sender
# -------------------------
def send_email_smtp(host, port, user, pw, sender, recipients, subject, body):
    try:
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(",") if r.strip()]
        msg = EmailMessage()
        msg["From"] = sender or user
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)
        server = smtplib.SMTP(host, port, timeout=20)
        if port == 587:
            server.starttls()
        server.login(user, pw)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as ex:
        return False, str(ex)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dashboard", 
    "🔎 Single Stock", 
    "💼 Portfolio", 
    "📣 Alerts", 
    "🧾 Watchlist Editor"
])

# -------------------------
# TAB 1: Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    wl = load_watchlist()
    if not wl:
        st.warning("Your watchlist is empty. Add stocks in the Watchlist Editor tab.")
    elif st.button("🔍 Run Analysis", type="primary"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(wl):
            info, _ = fetch_info_and_history(sym)
            if info.get("error"):
                continue
            ltp = safe_get(info, "currentPrice")
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)
            cap = rec["market_cap"]
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank = (rec["score"] * 2) + (rec["undervaluation_%"] or 0)/10 + cap_weight
            rows.append({
                "Symbol": sym,
                "LTP": ltp,
                "Fair Value": fv,
                "Underv%": rec["undervaluation_%"],
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec["recommendation"],
                "Score": rec["score"],
                "RankScore": round(rank, 2),
                "Reasons": "; ".join(rec["reasons"])
            })
            progress.progress(int(((i + 1) / len(wl)) * 100))

        df = pd.DataFrame(rows).sort_values(by="RankScore", ascending=False)

        def highlight(row):
            color = "#e8f5e9" if row["Rec"] in ["Buy", "Strong Buy"] else "#ffebee" if row["Rec"] == "Avoid" else ""
            return [f"background-color: {color}"] * len(row)

        st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# TAB 2: Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Detail")
    wl = load_watchlist()
    sel = st.selectbox("Select stock", wl) if wl else st.text_input("Enter symbol (e.g., RELIANCE)")
    if sel:
        info, hist = fetch_info_and_history(sel)
        if info.get("error"):
            st.error("Data fetch error: " + info["error"])
        else:
            ltp = safe_get(info, "currentPrice")
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)

            c1, c2, c3 = st.columns(3)
            c1.metric("LTP", f"₹{ltp:.2f}")
            c2.metric("Fair Value", f"₹{fv or '-'}")
            c3.metric("Recommendation", rec["recommendation"])

            st.markdown("### 📊 Fundamentals")
            st.json({
                "PE": safe_get(info, "trailingPE"),
                "EPS (TTM)": safe_get(info, "trailingEps"),
                "ROE": safe_get(info, "returnOnEquity"),
                "Debt/Equity": safe_get(info, "debtToEquity"),
                "Market Cap": safe_get(info, "marketCap"),
            })

            st.markdown("---")
            st.write(f"**Valuation Method:** {method}")
            st.write(f"**Buy Below:** ₹{buy} | **Sell Above:** ₹{sell}")
            st.write(f"**Undervaluation %:** {rec['undervaluation_%']}")
            st.info(f"Reasons: {', '.join(rec['reasons']) or '-'}")

            if not hist.empty:
                st.subheader("📈 5-Year Price Chart")
                st.line_chart(hist["Close"])
            else:
                st.info("No historical data available.")

# -------------------------
# TAB 3: Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV with columns: `symbol`, `buy_price`, `quantity`")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        pf = pd.read_csv(uploaded)
        pf.columns = [c.lower() for c in pf.columns]
        if not set(["symbol", "buy_price", "quantity"]).issubset(pf.columns):
            st.error("CSV must have columns: symbol, buy_price, quantity")
        else:
            rows = []
            for _, r in pf.iterrows():
                sym = str(r["symbol"]).upper().strip()
                buy, qty = float(r["buy_price"]), float(r["quantity"])
                info, _ = fetch_info_and_history(sym)
                ltp = safe_get(info, "currentPrice")
                current_val = ltp * qty if isinstance(ltp, (int, float)) else None
                invested = buy * qty
                pl = (current_val - invested) if current_val else None
                pl_pct = (pl / invested * 100) if pl else None
                rows.append({
                    "Symbol": sym,
                    "Buy Price": buy,
                    "Qty": qty,
                    "LTP": ltp,
                    "Invested": invested,
                    "Current Value": current_val,
                    "P/L": pl,
                    "P/L%": pl_pct,
                })
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.metric("Total P/L (₹)", f"{out['P/L'].sum(skipna=True):,.2f}")

# -------------------------
# TAB 4: Alerts
# -------------------------
with tab4:
    st.header("📣 Email Alerts")
    with st.form("alert_form"):
        smtp_host = st.text_input("SMTP Host", "smtp.gmail.com")
        smtp_port = st.number_input("Port", 587)
        smtp_user = st.text_input("Username (Email)")
        smtp_pass = st.text_input("Password (App Password Recommended)", type="password")
        sender = st.text_input("From", value=smtp_user)
        recipients = st.text_input("To (comma-separated)")
        threshold = st.number_input("Undervaluation% ≥", 10)
        submitted = st.form_submit_button("📨 Send Alerts Now")

    if submitted:
        wl = load_watchlist()
        results = []
        for sym in wl:
            info, _ = fetch_info_and_history(sym)
            if info.get("error"):
                continue
            ltp = safe_get(info, "currentPrice")
            fv, method = estimate_fair_value(info)
            if fv and ltp:
                underv = round(((fv - ltp) / fv) * 100, 2)
                if underv >= threshold:
                    results.append(f"{sym}: LTP ₹{ltp} | FV ₹{fv} | Underv {underv}% ({method})")
        if not results:
            st.info("No stocks crossed the threshold.")
        else:
            body = "\n".join(results)
            ok, msg = send_email_smtp(smtp_host, int(smtp_port), smtp_user, smtp_pass, sender, recipients, "StockMentor Alerts", body)
            st.success("✅ Alerts sent!") if ok else st.error("❌ " + msg)

# -------------------------
# TAB 5: Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Manage Watchlist")
    cur = load_watchlist()
    new_txt = st.text_area("Symbols (one per line, no .NS)", "\n".join(cur), height=250)
    if st.button("💾 Save Watchlist", use_container_width=True):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        st.success("✅ Saved!") if ok else st.error("❌ " + msg)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("✨ StockMentor — Rule-based, transparent & data-backed advisor (Yahoo Finance API)")
