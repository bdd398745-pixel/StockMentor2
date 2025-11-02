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
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and abs(roe) > 1:
        roe /= 100.0
    de = safe_get(info, "debtToEquity", np.nan)
    earnings_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)

    underval = None
    if fair_value and current_price and fair_value > 0:
        underval = round(((fair_value - current_price)/fair_value) * 100, 2)

    score = 0
    reasons = []

    # Quality (ROE)
    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 3; reasons.append("High ROE")
        elif roe >= 0.12:
            score += 2; reasons.append("Good ROE")
        elif roe > 0:
            score += 1; reasons.append("Positive ROE")

    # Debt
    if isinstance(de, (int, float)):
        if de <= 0.5:
            score += 2; reasons.append("Low D/E")
        elif de <= 1.5:
            score += 1; reasons.append("Moderate D/E")

    # Growth
    if isinstance(earnings_growth, (int, float)):
        if earnings_growth >= 0.20:
            score += 2; reasons.append("Strong earnings growth")
        elif earnings_growth >= 0.05:
            score += 1; reasons.append("Moderate earnings growth")

    # Valuation
    if isinstance(underval, (int, float)):
        if underval >= 25:
            score += 3; reasons.append("Deep undervaluation")
        elif underval >= 10:
            score += 2; reasons.append("Undervalued")
        elif underval >= 3:
            score += 1; reasons.append("Slight undervaluation")

    rec = "Hold"
    if score >= 7 and underval >= 10:
        rec = "Strong Buy"
    elif score >= 5 and underval >= 5:
        rec = "Buy"
    elif (isinstance(pe, (int, float)) and pe > 80) or (isinstance(roe, (int, float)) and roe < 0):
        rec = "Avoid"

    return {
        "score": score,
        "reasons": reasons,
        "undervaluation_%": underval,
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard",
    "🔎 Single Stock",
    "💼 Portfolio",
    "📣 Alerts",
    "🧾 Watchlist Editor",
    "🧠 Stock Screener"
])

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
# -------------------------
# 🔎 SINGLE STOCK TAB
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    st.caption("Analyze any Indian stock — rule-based insights with valuation, quality, growth & trend analysis.")

    symbol_input = st.text_input("Enter NSE symbol (e.g. HDFCBANK, TCS, RELIANCE):").strip().upper()

    if symbol_input:
        with st.spinner(f"Fetching data for {symbol_input}..."):
            info, hist = fetch_info_and_history(symbol_input)
            if info.get("error"):
                st.error(info["error"])
            elif hist.empty:
                st.warning("No price data found.")
            else:
                ltp = safe_get(info, "currentPrice", np.nan)
                fv, fv_method = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)
                roe = safe_get(info, "returnOnEquity", np.nan)
                if roe and abs(roe) > 1: roe /= 100
                de = safe_get(info, "debtToEquity", np.nan)
                growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
                market_cap = safe_get(info, "marketCap", np.nan)
                pe = safe_get(info, "trailingPE", np.nan)
                pb = safe_get(info, "priceToBook", np.nan)
                dy = safe_get(info, "dividendYield", np.nan)
                if dy and abs(dy) > 1: dy *= 100

                underval = rec["undervaluation_%"]
                cagr = None
                if fv and ltp and fv > 0 and ltp > 0:
                    try:
                        cagr = (fv/ltp)**(1/3) - 1
                        cagr = round(cagr*100, 2)
                    except:
                        cagr = None

                # --- Overview ---
                st.subheader("📘 Overview")
                c1, c2, c3 = st.columns(3)
                c1.metric("LTP (₹)", f"{ltp:,.2f}")
                c2.metric("Fair Value (₹)", f"{fv:,.2f}" if fv else "—", fv_method)
                if underval is not None:
                    c3.metric("Undervaluation %", f"{underval:.2f}%", delta_color="inverse")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Cap", f"{market_cap/1e9:,.1f} B" if market_cap else "—")
                c2.metric("PE", f"{pe:.1f}" if pe else "—")
                c3.metric("PB", f"{pb:.1f}" if pb else "—")
                c4.metric("Div Yield", f"{dy:.2f}%" if dy else "—")

                # --- Valuation Zone ---
                st.subheader("💰 Valuation Zone")
                if fv and ltp:
                    underval_pct = (fv - ltp)/fv * 100
                    progress_color = "green" if underval_pct > 0 else "red"
                    st.progress(min(max((100 - underval_pct), 0), 100)/100)
                    st.markdown(
                        f"**Fair Value:** ₹{fv:.2f} &nbsp;|&nbsp; **Current Price:** ₹{ltp:.2f} "
                        f"&nbsp;→&nbsp; **Undervaluation:** {underval_pct:.2f}%"
                    )
                st.write(f"**Buy Below:** ₹{buy} &nbsp;&nbsp; **Sell Above:** ₹{sell}")

                # --- Fundamental Strength ---
                st.subheader("📊 Fundamental Strength")
                strength_data = {
                    "Metric": ["ROE", "Debt/Equity", "Earnings Growth", "PE"],
                    "Value": [roe, de, growth, pe],
                    "Interpretation": [
                        "High profitability" if roe and roe >= 0.15 else "Low ROE" if roe else "N/A",
                        "Low leverage" if de and de < 0.5 else "Moderate" if de and de < 1.5 else "High debt",
                        "Strong growth" if growth and growth >= 0.2 else "Stable" if growth and growth >= 0 else "Decline",
                        "Reasonable valuation" if pe and pe < 25 else "Expensive" if pe else "N/A",
                    ],
                }
                st.table(pd.DataFrame(strength_data))

                # --- Trend Chart ---
                st.subheader("📈 Price Trend (1 Year)")
                try:
                    hist = hist.tail(252)  # approx 1 year
                    hist["50MA"] = hist["Close"].rolling(50).mean()
                    hist["200MA"] = hist["Close"].rolling(200).mean()
                    st.line_chart(hist[["Close", "50MA", "200MA"]])
                except Exception as e:
                    st.warning(f"Unable to render chart: {e}")

                # --- Scorecard (Safe version) ---
                st.subheader("⭐ Investment Scorecard")
                
                def safe_star(value):
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        return 0
                    return min(max(value, 0), 5)
                
                factors = {
                    "Quality (ROE)": safe_star((roe or 0) / 0.25 * 5 if roe else 0),
                    "Growth": safe_star((growth or 0) / 0.2 * 5 if growth else 0),
                    "Valuation": safe_star(5 if underval and underval >= 25 else 4 if underval and underval >= 10 else 3 if underval and underval >= 3 else 2),
                    "Risk (Low better)": safe_star(5 if de and de <= 0.5 else 3 if de and de <= 1.5 else 1),
                }
                
                score_df = pd.DataFrame({
                    "Factor": list(factors.keys()),
                    "Stars (out of 5)": [f"{int(round(v))} ⭐" for v in factors.values()]
                })
                st.table(score_df)


                # --- Interpretation ---
                st.subheader("🧠 Interpretation")
                interp = []
                interp.append(f"- Company shows **{'strong' if roe and roe>=0.15 else 'moderate' if roe and roe>0 else 'weak'} profitability** (ROE {roe*100:.1f}%)." if roe else "- ROE data unavailable.")
                interp.append(f"- Debt level is **{'low' if de and de<=0.5 else 'moderate' if de and de<=1.5 else 'high'}** (D/E {de:.2f})." if de else "- D/E data unavailable.")
                if growth and growth >= 0.2:
                    interp.append("- Earnings growth is strong; profit momentum intact.")
                elif growth and growth >= 0.05:
                    interp.append("- Moderate earnings growth; stable performance.")
                else:
                    interp.append("- Weak or negative earnings trend.")
                if underval and underval > 0:
                    interp.append(f"- Currently **undervalued by {underval:.1f}%** vs estimated fair value ₹{fv:.2f}.")
                if cagr:
                    interp.append(f"- Expected 3-year CAGR ≈ **{cagr:.1f}%** based on valuation gap.")
                interp.append(f"- Recommendation: **{rec['recommendation']}** based on total score {rec['score']}/10.")
                st.markdown("\n".join(interp))

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
# -------------------------
# TAB 6: 🧠 Stock Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Find Best Stocks Beyond Your Watchlist")
    st.caption("Automatically ranks Indian stocks using your rule-based model (ROE, D/E, growth, valuation).")

    # --- Universe selection ---
    source = st.radio(
        "Choose stock universe",
        ["NIFTY50", "NIFTY100", "NIFTY500", "Custom List"],
        horizontal=True
    )

    if source == "Custom List":
        custom_symbols = st.text_area("Enter comma-separated stock symbols (e.g. RELIANCE, TCS, HDFCBANK)")
        symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
    else:
        # small sample lists; expand as needed or load from file
        indices = {
            "NIFTY50": [
                "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","BHARTIARTL","KOTAKBANK"
            ],
            "NIFTY100": [
                "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","SUNPHARMA","ASIANPAINT",
                "AXISBANK","MARUTI","ULTRACEMCO","HINDUNILVR"
            ],
            "NIFTY500": [
                "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","SUNPHARMA","ASIANPAINT",
                "AXISBANK","MARUTI","ULTRACEMCO","HINDUNILVR","PERSISTENT","CIPLA","NTPC","ONGC","TATASTEEL","JSWSTEEL"
            ]
        }
        symbols = indices.get(source, [])

    st.info(f"Loaded {len(symbols)} symbols for screening.")

    # Filter panel
    st.markdown("**Filters (optional)**")
    col_f1, col_f2, col_f3 = st.columns(3)
    min_roe = col_f1.number_input("Min ROE (%)", value=0.0, step=0.5)
    max_de = col_f2.number_input("Max Debt/Equity", value=2.0, step=0.1)
    min_underv = col_f3.number_input("Min Undervaluation %", value=0.0, step=1.0)

    run = st.button("🚀 Run Screener")
    if run:
        if not symbols:
            st.warning("No symbols to screen. Choose Custom List or a predefined index.")
        else:
            progress = st.progress(0)
            rows = []
            for i, sym in enumerate(symbols):
                info, _ = fetch_info_and_history(sym)
                # skip when fetching failed
                if not info or info.get("error"):
                    progress.progress(int(((i+1)/len(symbols))*100))
                    continue

                ltp = safe_get(info, "currentPrice", np.nan)
                fv, fv_method = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)

                # apply simple filters: skip if fails
                roe = safe_get(info, "returnOnEquity", np.nan)
                if roe and abs(roe) > 1:
                    roe = roe/100.0
                de = safe_get(info, "debtToEquity", np.nan)
                underv = rec.get("undervaluation_%")
                if (not pd.isna(roe) and (roe*100) < min_roe) or (not pd.isna(de) and de > max_de) or (not pd.isna(underv) and underv < min_underv):
                    progress.progress(int(((i+1)/len(symbols))*100))
                    continue

                cap = rec.get("market_cap")
                cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
                rank_score = (rec["score"] * 2) + ((underv or 0) / 10) + cap_weight

                rows.append({
                    "Symbol": sym,
                    "LTP": round(ltp,2) if isinstance(ltp,(int,float)) and not math.isnan(ltp) else np.nan,
                    "Fair Value": fv,
                    "Underv%": underv,
                    "Valuation Method": fv_method,
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"],
                    "RankScore": round(rank_score, 2),
                    "Reasons": "; ".join(rec.get("reasons", []))
                })
                progress.progress(int(((i+1)/len(symbols))*100))
                time.sleep(0.05)

            df = pd.DataFrame(rows)
            if df.empty:
                st.warning("No stocks passed the filters or no valid data found.")
            else:
                df_sorted = df.sort_values(by="RankScore", ascending=False).reset_index(drop=True)
                # Show color hint for top picks
                st.dataframe(df_sorted, use_container_width=True)

                st.subheader("🏆 Top 5 Picks")
                st.table(df_sorted.head(5)[["Symbol","Rec","Score","Underv%","Buy Below","Fair Value"]])

                csv = df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Screener Results", csv, "screener_results.csv", "text/csv")
