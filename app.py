# app.py
"""
StockMentor - Enhanced Rule-based long-term stock analyst (India)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, Stock Screener (Enhanced)
- No LLMs; purely quantitative + rule-based scoring
- Adds live NIFTY index fetching, technical momentum, multi-factor ranking, & advanced filters
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="StockMentor", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Scientific Rule-based Stock Advisor (India)")
st.caption("Enhanced model: combines fundamentals + valuation + momentum for intelligent ranking")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0

# -------------------------
# Utility Functions
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().tolist()
    except Exception:
        return []

def save_watchlist(symbols):
    pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
    return True

@st.cache_data(ttl=1200)
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period="6mo", interval="1d")
        return info, hist
    except Exception:
        return {}, pd.DataFrame()

def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    return default if v in ("None", "", None) else v

# -------------------------
# Fair Value Estimation
# -------------------------
def estimate_fair_value(info):
    eps = safe_get(info, "trailingEps", np.nan)
    pe = safe_get(info, "trailingPE", DEFAULT_PE_TARGET)
    if isinstance(eps, (int, float)) and eps > 0:
        return round(eps * pe, 2)
    return np.nan

def compute_buy_sell(fv, mos=0.25):
    if not fv or math.isnan(fv):
        return None, None
    return round(fv * (1 - mos), 2), round(fv * (1 + mos/1.5), 2)

# -------------------------
# Rule-based Recommendation
# -------------------------
def rule_based_recommendation(info, fv, price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and roe > 1: roe /= 100
    de = safe_get(info, "debtToEquity", np.nan)
    growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)
    underv = ((fv - price)/fv)*100 if fv and price and fv>0 else np.nan

    score = 0
    if roe and roe >= 0.20: score += 3
    elif roe and roe >= 0.10: score += 2
    if de and de <= 0.5: score += 2
    elif de and de <= 1.5: score += 1
    if growth and growth >= 0.15: score += 2
    elif growth and growth >= 0.05: score += 1
    if underv and underv >= 20: score += 3
    elif underv and underv >= 10: score += 2

    rec = "Hold"
    if score >= 7: rec = "Buy"
    elif score >= 5: rec = "Consider"
    elif de and de > 2: rec = "Avoid"
    return {"score": score, "rec": rec, "underv": underv, "roe": roe, "de": de, "growth": growth, "mcap": market_cap}

# -------------------------
# Technical Momentum
# -------------------------
def get_momentum(hist):
    if hist.empty or "Close" not in hist:
        return "N/A"
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()
    last = hist.iloc[-1]
    if last["Close"] > last["MA200"]:
        return "Bullish"
    elif last["Close"] > last["MA50"]:
        return "Neutral"
    else:
        return "Bearish"

# -------------------------
# Fetch NIFTY constituents
# -------------------------
@st.cache_data(ttl=3600)
def get_index_list(name="NIFTY50"):
    urls = {
        "NIFTY50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "NIFTY100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "NIFTY200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "NIFTY500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    }
    try:
        df = pd.read_csv(urls[name])
        return df["Symbol"].tolist()
    except Exception:
        return []

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Stock Screener"
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
# TAB 6: Enhanced Screener
# -------------------------
with tab6:
    st.header("🧠 Enhanced Stock Screener")
    st.caption("Ranks stocks scientifically using fundamentals, valuation & momentum.")

    source = st.radio("Select Stock Universe", ["NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500", "Custom"], horizontal=True)
    if source == "Custom":
        custom_symbols = st.text_area("Enter comma-separated symbols").split(",")
        symbols = [s.strip().upper() for s in custom_symbols if s.strip()]
    else:
        symbols = get_index_list(source)

    st.info(f"{len(symbols)} symbols loaded for screening.")

    col1, col2, col3, col4 = st.columns(4)
    min_roe = col1.number_input("Min ROE %", value=0.0)
    max_de = col2.number_input("Max D/E", value=2.0)
    min_underv = col3.number_input("Min Undervaluation %", value=0.0)
    momentum_filter = col4.selectbox("Momentum", ["Any", "Bullish", "Neutral", "Bearish"])

    run = st.button("🚀 Run Screener")

    if run:
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(symbols):
            info, hist = fetch_info_and_history(sym)
            if not info: continue
            price = safe_get(info, "currentPrice", np.nan)
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            mom = get_momentum(hist)

            # Apply filters
            if rec["roe"] and rec["roe"]*100 < min_roe: continue
            if rec["de"] and rec["de"] > max_de: continue
            if rec["underv"] and rec["underv"] < min_underv: continue
            if momentum_filter != "Any" and mom != momentum_filter: continue

            # Weighted ranking
            q = min(max(rec["roe"]*100/25, 0), 5)
            g = min(max((rec["growth"] or 0)/0.2*5, 0), 5)
            v = min(max((rec["underv"] or 0)/20*5, 0), 5)
            m = 5 if mom == "Bullish" else 3 if mom == "Neutral" else 1
            size = 5 if rec["mcap"] and rec["mcap"]>5e11 else 3 if rec["mcap"] and rec["mcap"]>1e11 else 1
            final_score = round(0.3*q + 0.2*g + 0.25*v + 0.15*m + 0.1*size, 2)

            # sparkline data
            if not hist.empty:
                recent = hist["Close"].tail(30).tolist()
            else:
                recent = []

            rows.append({
                "Symbol": sym,
                "LTP": price,
                "Fair Value": fv,
                "Underv%": rec["underv"],
                "ROE%": round((rec["roe"] or 0)*100,2),
                "D/E": rec["de"],
                "Growth": rec["growth"],
                "Momentum": mom,
                "Recommendation": rec["rec"],
                "Score": rec["score"],
                "FinalScore": final_score,
                "Trend": recent
            })
            progress.progress((i+1)/len(symbols))
            time.sleep(0.05)

        if not rows:
            st.warning("No stocks passed filters.")
        else:
            df = pd.DataFrame(rows)
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
            st.success("✅ Screening Complete — ranked scientifically.")
            st.dataframe(df.drop(columns=["Trend"]), use_container_width=True)

            st.subheader("🏆 Top 5 Stocks")
            st.table(df.head(5)[["Symbol","Recommendation","Underv%","ROE%","Momentum","FinalScore"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Screener Results", csv, "enhanced_screener.csv", "text/csv")
