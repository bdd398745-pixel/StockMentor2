# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor
- Rule-based scoring & recommendation
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
st.caption("No OpenAI. Pure rule-based valuation and recommendations.")

# -------------------------
# Constants & settings
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0  # fallback PE target for fair-value calc
DISCOUNT_RATE = 0.10      # used for simple DCF-like fallback
MOCK_SLEEP = 0.02         # small pause when looping for UX (not required)

# -------------------------
# Utility: Watchlist load/save
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        # Normalize: remove empty, remove .NS suffix if present (we'll add .NS later)
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
        # Clear streamlit cache for load_watchlist to reflect change
        load_watchlist.clear()
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Data fetchers & helpers
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol_no_suffix):
    """
    Returns (info_dict, history_df)
    symbol_no_suffix: e.g., 'INFY' or 'RELIANCE'
    We will query symbol + '.NS' for NSE tickers
    """
    symbol = f"{symbol_no_suffix}.NS"
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        # fetch 5y monthly history for trend and basic revenue/growth estimates if needed
        hist = ticker.history(period="5y", interval="1d")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    if v in (None, "None", ""):
        return default
    return v

# -------------------------
# Fair value calculation (rule-based)
# Priority:
# 1) Use analyst targetMeanPrice if present & reasonable
# 2) EPS * fallback PE (choose forwardPE/trailingPE or default)
# 3) BookValue * trailingPE (fallback)
# -------------------------
def estimate_fair_value(info):
    # try analyst target
    try:
        target = safe_get(info, "targetMeanPrice", np.nan)
        if isinstance(target, (int, float)) and not math.isnan(target) and target > 0:
            return round(float(target), 2), "AnalystTarget"
    except Exception:
        pass

    # get EPS and select PE target
    eps = safe_get(info, "trailingEps", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    trailing_pe = safe_get(info, "trailingPE", np.nan)

    # choose pe_target heuristically
    pe_target = None
    if isinstance(forward_pe, (int, float)) and not math.isnan(forward_pe) and forward_pe > 0 and forward_pe < 200:
        pe_target = forward_pe
    elif isinstance(trailing_pe, (int, float)) and not math.isnan(trailing_pe) and trailing_pe > 0 and trailing_pe < 200:
        pe_target = max(10.0, trailing_pe * 0.9)  # slightly conservative
    else:
        pe_target = DEFAULT_PE_TARGET

    if isinstance(eps, (int, float)) and not math.isnan(eps) and eps > 0:
        fv = eps * pe_target
        return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"

    # fallback: bookValue * trailingPE
    book = safe_get(info, "bookValue", np.nan)
    if isinstance(book, (int, float)) and not math.isnan(book) and isinstance(trailing_pe, (int, float)) and not math.isnan(trailing_pe) and trailing_pe > 0:
        fv = book * trailing_pe
        return round(float(fv), 2), "BVxPE"

    # if nothing computable
    return None, "InsufficientData"

# -------------------------
# Buy / Sell zones & MOS
# -------------------------
def compute_buy_sell(fair_value, mos=0.30):
    if fair_value is None or math.isnan(fair_value):
        return None, None
    buy_price = round(fair_value * (1 - mos), 2)
    sell_price = round(fair_value * (1 + (mos/1.5)), 2)  # a more moderate sell target
    return buy_price, sell_price

# -------------------------
# Rule-based scoring / recommendation
# Use ROE, debtToEquity, revenue/earnings growth if available, undervaluation %
# -------------------------
def rule_based_recommendation(info, fair_value, current_price):
    # collect metrics
    try:
        roe = safe_get(info, "returnOnEquity", np.nan)  # often in decimal (0.12 = 12%)
        if roe and abs(roe) > 1:  # sometimes yfinance returns % instead of decimal
            # if >1 assume it's percent and convert
            roe = roe / 100.0
        de = safe_get(info, "debtToEquity", np.nan)  # ratio
        eps = safe_get(info, "trailingEps", np.nan)
        earnings_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)  # quarter-on-quarter
        market_cap = safe_get(info, "marketCap", np.nan)
        pe = safe_get(info, "trailingPE", np.nan)
    except Exception:
        roe = de = eps = earnings_growth = market_cap = pe = np.nan

    # undervaluation %
    underval = None
    if fair_value and current_price and fair_value > 0:
        underval = round(((fair_value - current_price)/fair_value) * 100, 2)

    # Score components (0-3 each)
    score = 0
    reasons = []

    # Quality: ROE
    if isinstance(roe, (int, float)) and not math.isnan(roe):
        if roe >= 0.20:
            score += 3; reasons.append("High ROE")
        elif roe >= 0.12:
            score += 2; reasons.append("Good ROE")
        elif roe > 0:
            score += 1; reasons.append("Positive ROE")
        else:
            reasons.append("Negative/Low ROE")

    # Leverage: Debt-to-Equity (lower better)
    if isinstance(de, (int, float)) and not math.isnan(de):
        if de <= 0.5:
            score += 2; reasons.append("Low D/E")
        elif de <= 1.5:
            score += 1; reasons.append("Moderate D/E")
        else:
            reasons.append("High D/E")

    # Growth: EPS / earnings growth
    if isinstance(earnings_growth, (int, float)) and not math.isnan(earnings_growth):
        # earningsQuarterlyGrowth is usually decimal (0.10 = 10%)
        if earnings_growth >= 0.20:
            score += 2; reasons.append("Strong recent earnings growth")
        elif earnings_growth >= 0.05:
            score += 1; reasons.append("Moderate earnings growth")

    # Valuation boost if undervalued materially
    if isinstance(underval, (int, float)):
        if underval >= 25:
            score += 3; reasons.append("Deep undervaluation")
        elif underval >= 10:
            score += 2; reasons.append("Undervalued")
        elif underval >= 3:
            score += 1; reasons.append("Slight undervaluation")
        elif underval < -10:
            reasons.append("Overvalued")

    # Market cap consideration (prefer bigger caps for long-term stability)
    if isinstance(market_cap, (int, float)) and not math.isnan(market_cap):
        if market_cap >= 10_000_000_000:  # >10k crore (approx) but currency in INR? yfinance marketCap for Indian tickers is in INR (usually)
            score += 1
            reasons.append("Large market cap")

    # Final recommendation from score + hard checks
    rec = "Hold"
    if score >= 7 and (isinstance(underval, (int, float)) and underval >= 10):
        rec = "Strong Buy"
    elif score >= 5 and (isinstance(underval, (int, float)) and underval >= 5):
        rec = "Buy"
    else:
        # check for red flags
        if (isinstance(pe, (int, float)) and pe and pe > 80) or (isinstance(roe, (int, float)) and roe < 0):
            rec = "Avoid"
            reasons.append("High PE or negative ROE")
        else:
            rec = "Hold"

    return {
        "score": score,
        "reasons": reasons,
        "undervaluation_%": underval,
        "recommendation": rec
    }

# -------------------------
# Email helper (simple)
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
        server.ehlo()
        if smtp_port == 587:
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as ex:
        return False, str(ex)

# -------------------------
# UI: Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts (Email)", "🧾 Watchlist Editor"])

# -------------------------
# TAB: Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist is empty. Go to Watchlist Editor to add symbols (one per line).")
    else:
        if st.button("🔍 Analyze watchlist now"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, hist = fetch_info_and_history(sym)
                time.sleep(MOCK_SLEEP)
                if info.get("error"):
                    rows.append({
                        "Symbol": sym,
                        "LTP": None,
                        "Fair Value": None,
                        "Underv%": None,
                        "Rec": f"Error: {info.get('error')}"
                    })
                    progress.progress(int(((i+1)/len(watchlist))*100))
                    continue

                ltp = safe_get(info, "currentPrice", np.nan)
                fv, method = estimate_fair_value(info)
                buy, sell = (None, None)
                if fv:
                    buy, sell = compute_buy_sell(fv)
                rec_info = rule_based_recommendation(info, fv, ltp)
                rows.append({
                    "Symbol": sym,
                    "Name": safe_get(info, "shortName", sym),
                    "LTP": round(ltp,2) if isinstance(ltp, (int, float)) and not math.isnan(ltp) else None,
                    "Fair Value": fv,
                    "Valuation Method": method,
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Underv%": rec_info.get("undervaluation_%"),
                    "Rec": rec_info.get("recommendation"),
                    "Score": rec_info.get("score"),
                    "Reasons": "; ".join(rec_info.get("reasons", []))
                })
                progress.progress(int(((i+1)/len(watchlist))*100))
            df = pd.DataFrame(rows)
            # Reorder columns for nicer display
            cols = ["Symbol","Name","LTP","Fair Value","Valuation Method","Underv%","Buy Below","Sell Above","Rec","Score","Reasons"]
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols].sort_values(by="Underv%", ascending=False, na_position="last"), use_container_width=True)
        else:
            st.info("Click 'Analyze watchlist now' to fetch fundamentals and recommendations.")

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
