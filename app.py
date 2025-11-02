# app.py
"""
StockMentor - Rule-based Long-Term Stock Analyst (India)
Enhanced with Financial Modeling Prep (FMP) API Support
- Optional FMP API (toggle)
- yfinance fallback
- Screener for Nifty 50 / Nifty 500
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import math
from datetime import datetime

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (Enhanced)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Now with FMP API support & Nifty 500 Screener 🚀")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
DISCOUNT_RATE = 0.10

# Load API Key
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
if FMP_API_KEY:
    st.sidebar.success("✅ FMP key detected")
else:
    st.sidebar.error("❌ No FMP key found — using yfinance fallback")

# -------------------------
# Load Nifty Stock List (Predefined)
# -------------------------
@st.cache_data
def load_nifty_stocks():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        return df[["Symbol", "Company Name", "Industry"]]
    except Exception as e:
        st.error(f"Could not load Nifty500 list: {e}")
        return pd.DataFrame(columns=["Symbol", "Company Name", "Industry"])

# -------------------------
# Watchlist Load/Save
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().tolist()
    except:
        return []

def save_watchlist(symbols):
    pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
    load_watchlist.clear()

# -------------------------
# FMP Fetcher
# -------------------------
def fetch_fmp_data(symbol):
    try:
        base = "https://financialmodelingprep.com/api/v3"
        quote_url = f"{base}/quote/{symbol}.NS?apikey={FMP_API_KEY}"
        profile_url = f"{base}/profile/{symbol}.NS?apikey={FMP_API_KEY}"
        hist_url = f"{base}/historical-price-full/{symbol}.NS?timeseries=365&apikey={FMP_API_KEY}"

        quote = requests.get(quote_url).json()
        profile = requests.get(profile_url).json()
        hist = requests.get(hist_url).json()

        if not quote or not profile:
            return {}, pd.DataFrame()

        q = quote[0] if isinstance(quote, list) else quote
        p = profile[0] if isinstance(profile, list) else profile

        info = {
            "symbol": symbol,
            "currentPrice": q.get("price"),
            "marketCap": q.get("marketCap"),
            "trailingPE": q.get("pe"),
            "priceToBook": q.get("pb"),
            "dividendYield": (q.get("lastDiv") or 0) / q.get("price") if q.get("price") else 0,
            "companyName": p.get("companyName"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "beta": p.get("beta"),
            "returnOnEquity": p.get("returnOnEquityTTM"),
            "debtToEquity": p.get("debtToEquityTTM"),
            "eps": p.get("epsTTM")
        }

        if "historical" in hist:
            df = pd.DataFrame(hist["historical"])
            df = df.rename(columns={"date": "Date", "close": "Close"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        else:
            df = pd.DataFrame()

        return info, df

    except Exception as e:
        st.warning(f"FMP fetch error for {symbol}: {e}")
        return {}, pd.DataFrame()

# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# Utility Functions
# -------------------------
def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    return default if v in (None, "None", "") else v

def estimate_fair_value(info):
    eps = safe_get(info, "eps", safe_get(info, "trailingEps", np.nan))
    pe = safe_get(info, "trailingPE", np.nan)
    if isinstance(eps, (int, float)) and eps > 0:
        pe_target = pe if isinstance(pe, (int, float)) and pe > 0 else DEFAULT_PE_TARGET
        fv = eps * pe_target
        return round(fv, 2)
    return np.nan

def rule_based_recommendation(info, fair_value, price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    de = safe_get(info, "debtToEquity", np.nan)
    underval = None if not fair_value or not price else round(((fair_value - price) / fair_value) * 100, 2)
    score = 0
    if isinstance(roe, (int, float)):
        if roe >= 0.20: score += 3
        elif roe >= 0.12: score += 2
    if isinstance(de, (int, float)):
        if de <= 0.5: score += 2
        elif de <= 1.5: score += 1
    if isinstance(underval, (int, float)):
        if underval >= 25: score += 3
        elif underval >= 10: score += 2
    rec = "Hold"
    if score >= 7: rec = "Strong Buy"
    elif score >= 5: rec = "Buy"
    return {"score": score, "recommendation": rec, "undervaluation": underval}

# -------------------------
# Tabs
# -------------------------
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
# Single Stock
# -------------------------
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ----------------------------------------------------
# Load FMP API key securely from Streamlit secrets
# ----------------------------------------------------
FMP_API_KEY = st.secrets["FMP_API_KEY"]

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="StockMentor - Single Stock Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📊 StockMentor — Single Stock Analysis")
st.caption("Powered by Financial Modeling Prep (FMP) API")

# ----------------------------------------------------
# Data Fetch Functions
# ----------------------------------------------------
@st.cache_data(ttl=1200)
def fetch_info_and_history(symbol):
    """Fetch company info and last 6 months price history from FMP"""
    try:
        # --- Company Info ---
        info_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
        info_data = requests.get(info_url).json()
        if not info_data or isinstance(info_data, dict):
            return {}, pd.DataFrame()
        info = info_data[0]

        # --- Historical Prices ---
        hist_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&timeseries=180&apikey={FMP_API_KEY}"
        hist_data = requests.get(hist_url).json()
        if "historical" not in hist_data:
            return info, pd.DataFrame()
        hist_df = pd.DataFrame(hist_data["historical"])
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.sort_values("date")
        hist_df.rename(columns={"close": "Close"}, inplace=True)

        return info, hist_df
    except Exception as e:
        st.error(f"❌ Data fetch error: {e}")
        return {}, pd.DataFrame()


@st.cache_data(ttl=1200)
def fetch_fundamentals(symbol):
    """Fetch key financial metrics"""
    try:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?limit=1&apikey={FMP_API_KEY}"
        data = requests.get(url).json()
        if not data or isinstance(data, dict):
            return {}
        return data[0]
    except Exception as e:
        st.error(f"❌ Fundamental fetch error: {e}")
        return {}

# ----------------------------------------------------
# User Input
# ----------------------------------------------------
symbol = st.text_input("🔍 Enter Stock Symbol (e.g. AAPL, TSLA, MSFT, TCS.NS):", "AAPL").upper()

if symbol:
    with st.spinner("Fetching data from FMP..."):
        info, hist = fetch_info_and_history(symbol)
        fundamentals = fetch_fundamentals(symbol)

    # ----------------------------------------------------
    # Company Info
    # ----------------------------------------------------
    if info:
        st.subheader(f"{info.get('companyName', symbol)} ({info.get('symbol', '')})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${info.get('price', 'N/A')}")
            st.metric("P/E Ratio", info.get("pe", "N/A"))
        with col2:
            st.metric("Market Cap", f"${info.get('mktCap', 0):,}")
            st.metric("Beta", info.get("beta", "N/A"))
        with col3:
            st.metric("Industry", info.get("industry', 'N/A"))
            st.metric("Exchange", info.get("exchangeShortName', 'N/A"))

        st.markdown(f"[🌐 Visit Website]({info.get('website', '#')})")

    else:
        st.warning("No company information found for this symbol.")

    # ----------------------------------------------------
    # Chart
    # ----------------------------------------------------
    if not hist.empty:
        st.write("### 📈 6-Month Price Trend")
        st.line_chart(hist.set_index("date")["Close"])
    else:
        st.warning("No price history available.")

    # ----------------------------------------------------
    # Fundamentals
    # ----------------------------------------------------
    if fundamentals:
        st.write("### 💡 Key Financial Metrics")
        metrics_df = pd.DataFrame([
            ["ROE (TTM)", fundamentals.get("roeTTM", "N/A")],
            ["ROA (TTM)", fundamentals.get("roaTTM", "N/A")],
            ["Debt/Equity (TTM)", fundamentals.get("debtToEquityTTM", "N/A")],
            ["Current Ratio (TTM)", fundamentals.get("currentRatioTTM", "N/A")],
            ["P/B Ratio (TTM)", fundamentals.get("pbRatioTTM", "N/A")],
            ["Dividend Yield (TTM)", fundamentals.get("dividendYieldTTM", "N/A")],
            ["Free Cash Flow Yield (TTM)", fundamentals.get("freeCashFlowYieldTTM", "N/A")]
        ], columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Fundamental data not available for this stock.")
else:
    st.info("Enter a stock symbol above to begin.")



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
# Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Find Best Stocks from Nifty 500")
    use_fmp = st.toggle("Use FMP API (recommended)", value=bool(FMP_API_KEY))
    nifty = load_nifty_stocks()

    if st.button("Run Screener"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(nifty["Symbol"].tolist()[:100]):  # limit to 100 for demo
            info, _ = fetch_fmp_data(sym) if use_fmp else fetch_info_and_history(sym)
            price = safe_get(info, "currentPrice", np.nan)
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": sym,
                "LTP": price,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
            progress.progress(int(((i + 1) / len(nifty[:100])) * 100))
        df = pd.DataFrame(rows)
        df = df.sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.success("✅ Screener completed and ranked by score")

st.caption("Made by Biswanath 🔍 | Rule-based, API-optional, Fully Offline-capable")
