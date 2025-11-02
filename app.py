# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Twelve Data API integration (primary) + yfinance fallback
Features preserved: Watchlist, Single Stock, Portfolio, Alerts, Watchlist Editor, Screener
Author: Biswanath (adapted)
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
st.set_page_config(page_title="StockMentor (TwelveData)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Twelve Data primary (with yfinance fallback) — Rule-based, API-optional, Fully Offline-capable")

# -------------------------
# Constants & Files
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 15.0  # used in fair value heuristics if needed

# Load API Keys from Streamlit secrets or env
TWELVE_API_KEY = st.secrets.get("TWELVE_API_KEY", os.getenv("TWELVE_API_KEY", ""))
# (No FMP anymore) yfinance is used as fallback automatically

# -------------------------
# Utilities / Caching
# -------------------------
@st.cache_data
def load_nifty_stocks():
    """Load Nifty500 list from NSE archives (best-effort)."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        # Keep some columns if present
        cols = [c for c in ["Symbol", "Company Name", "Industry"] if c in df.columns]
        return df[cols]
    except Exception:
        # return empty but consistent dataframe
        return pd.DataFrame(columns=["Symbol", "Company Name", "Industry"])

@st.cache_data
def load_watchlist():
    """Return list of symbols from WATCHLIST_FILE; one per line."""
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().str.upper().tolist()
    except Exception:
        return []

def save_watchlist(symbols):
    """Save list of symbols to WATCHLIST_FILE."""
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        # clear the cached loader so next load picks up changes
        try:
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)

def safe_get(info, key, default=np.nan):
    """Safe get for various info dict shapes."""
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if v in (None, "None", ""):
        return default
    return v

# -------------------------
# Valuation & Rules
# -------------------------
def estimate_fair_value(info):
    """
    Estimate fair value using multiple heuristics:
    - EPS * 15 (preferred)
    - Price / PE * targetPE (P/E reversion)
    - P/B * price (heuristic)
    """
    try:
        price = float(safe_get(info, "price", np.nan) or safe_get(info, "currentPrice", np.nan) or np.nan)
        # PE can be under different keys
        pe = np.nan
        for k in ("trailingPE", "pe", "pe_ratio", "peRatio"):
            v = safe_get(info, k, np.nan)
            try:
                pe = float(v) if not pd.isna(v) else pe
                if not pd.isna(pe):
                    break
            except Exception:
                continue

        eps = np.nan
        for k in ("eps", "epsTTM", "trailingEps", "basic_eps", "earnings_per_share"):
            v = safe_get(info, k, np.nan)
            try:
                eps = float(v) if not pd.isna(v) else eps
                if not pd.isna(eps):
                    break
            except Exception:
                continue

        pb = np.nan
        for k in ("priceToBook", "pb", "pb_ratio", "pb_ratio_mrq"):
            v = safe_get(info, k, np.nan)
            try:
                pb = float(v) if not pd.isna(v) else pb
                if not pd.isna(pb):
                    break
            except Exception:
                continue

        # Roe & other metrics
        roe = np.nan
        for k in ("returnOnEquity", "roe", "roeTTM"):
            v = safe_get(info, k, np.nan)
            try:
                roe = float(v) if not pd.isna(v) else roe
                if not pd.isna(roe):
                    break
            except Exception:
                continue

        if not pd.isna(eps) and eps > 0:
            fair_value = eps * DEFAULT_PE_TARGET
            method = "EPS-based"
        elif not pd.isna(pe) and pe > 0 and not pd.isna(price):
            # revert to DEFAULT_PE_TARGET
            fair_value = (price / pe) * DEFAULT_PE_TARGET
            method = "P/E reversion"
        elif not pd.isna(pb) and not pd.isna(price):
            fair_value = pb * price
            method = "P/B heuristic"
        else:
            fair_value = price
            method = "fallback"

        return round(float(fair_value), 2) if not pd.isna(fair_value) else np.nan, method
    except Exception as e:
        print(f"estimate_fair_value() error: {e}")
        return np.nan, "error"

def compute_buy_sell(fv, mos=0.25):
    """Simple margin of safety buy / target sell calculation."""
    try:
        fv = float(fv)
        if math.isnan(fv) or fv <= 0:
            return None, None
        buy = round(fv * (1 - mos), 2)
        sell = round(fv * (1 + mos / 1.5), 2)
        return buy, sell
    except Exception as e:
        print(f"compute_buy_sell() error: {e}")
        return None, None

def rule_based_recommendation(info, fair_value, price):
    """
    Score using a few factors: ROE, Debt/Equity, undervaluation.
    Return score and short recommendation.
    """
    try:
        roe = safe_get(info, "returnOnEquity", np.nan)
        if isinstance(roe, str):
            try:
                roe = float(roe)
            except:
                roe = np.nan
        de = safe_get(info, "debtToEquity", np.nan)
        if isinstance(de, str):
            try:
                de = float(de)
            except:
                de = np.nan

        underval = None
        try:
            if fair_value and price and not pd.isna(fair_value) and not pd.isna(price) and fair_value != 0:
                underval = round(((fair_value - price) / fair_value) * 100, 2)
        except Exception:
            underval = None

        score = 0
        if isinstance(roe, (int, float)) and not pd.isna(roe):
            if roe >= 0.20:
                score += 3
            elif roe >= 0.12:
                score += 2
        if isinstance(de, (int, float)) and not pd.isna(de):
            if de <= 0.5:
                score += 2
            elif de <= 1.5:
                score += 1
        if isinstance(underval, (int, float)):
            if underval >= 25:
                score += 3
            elif underval >= 10:
                score += 2

        rec = "Hold"
        if score >= 7:
            rec = "Strong Buy"
        elif score >= 5:
            rec = "Buy"
        elif score <= 2:
            rec = "Avoid / Monitor"

        return {"score": score, "recommendation": rec, "undervaluation": underval}
    except Exception as e:
        print(f"rule_based_recommendation() error: {e}")
        return {"score": 0, "recommendation": "Hold", "undervaluation": None}

# -------------------------
# Twelve Data Fetcher (primary)
# -------------------------
@st.cache_data(ttl=900)
def fetch_twelvedata_data(symbol: str):
    """
    Fetch stock info and 1-year history using Twelve Data API.
    Returns (info_dict, hist_df). If failure, returns ({}, pd.DataFrame()).
    """
    if not TWELVE_API_KEY:
        return {}, pd.DataFrame()

    try:
        base = "https://api.twelvedata.com"
        symbol_full = f"{symbol}.NS"

        # Quote (current price & basic)
        quote_url = f"{base}/quote?symbol={symbol_full}&apikey={TWELVE_API_KEY}"
        # Fundamentals (may include valuation ratios, earnings, financials)
        fundamentals_url = f"{base}/fundamentals?symbol={symbol_full}&apikey={TWELVE_API_KEY}"
        # Historical time series (daily)
        ts_url = f"{base}/time_series?symbol={symbol_full}&interval=1day&outputsize=365&format=JSON&apikey={TWELVE_API_KEY}"

        q_resp = requests.get(quote_url, timeout=10).json()
        f_resp = requests.get(fundamentals_url, timeout=10).json()
        ts_resp = requests.get(ts_url, timeout=15).json()

        # Basic validation
        if (not isinstance(q_resp, dict)) or ("price" not in q_resp and "close" not in q_resp):
            # Twelve returns {"status":"error", "message": "..."} on error
            raise Exception(q_resp.get("message", "No quote data from Twelve Data"))

        # Build info dict mapping to our expected keys as best-effort
        info = {}
        # price / currentPrice
        price_val = q_resp.get("price") or q_resp.get("close") or q_resp.get("previous_close")
        try:
            info["price"] = float(price_val) if price_val is not None else np.nan
        except Exception:
            info["price"] = np.nan

        # map some fundamentals if present
        # Twelve's fundamentals JSON is nested; attempt to extract commonly useful values:
        # Example keys: valuation_ratios -> pe_ratio, pb_ratio, dividend_yield
        try:
            # Many fields may be None or missing; use .get carefully
            if isinstance(f_resp, dict):
                # Company name
                info["companyName"] = f_resp.get("name") or f_resp.get("company_name") or symbol
                # valuations
                vr = f_resp.get("valuation_ratios", {}) or {}
                info["trailingPE"] = float(vr.get("pe_ratio")) if vr.get("pe_ratio") not in (None, "") else np.nan
                info["priceToBook"] = float(vr.get("pb_ratio")) if vr.get("pb_ratio") not in (None, "") else np.nan
                # dividend yield sometimes provided as 'dividend_yield' string like '1.23%'
                dy = vr.get("dividend_yield") or vr.get("dividendYield")
                if isinstance(dy, str) and "%" in dy:
                    try:
                        info["dividendYield"] = float(dy.strip().replace("%", "")) / 100.0
                    except:
                        info["dividendYield"] = np.nan
                else:
                    try:
                        info["dividendYield"] = float(dy) if dy not in (None, "") else np.nan
                    except:
                        info["dividendYield"] = np.nan

                # EPS
                eps_obj = f_resp.get("earnings_per_share", {}) or {}
                try:
                    info["eps"] = float(eps_obj.get("basic_eps")) if eps_obj.get("basic_eps") not in (None, "") else np.nan
                except:
                    info["eps"] = np.nan

                # profitability / roe
                prof = f_resp.get("profitability", {}) or {}
                try:
                    info["returnOnEquity"] = float(prof.get("roe")) if prof.get("roe") not in (None, "") else np.nan
                except:
                    info["returnOnEquity"] = np.nan

                # financial health -> debt_to_equity
                fh = f_resp.get("financial_health", {}) or {}
                try:
                    info["debtToEquity"] = float(fh.get("debt_to_equity")) if fh.get("debt_to_equity") not in (None, "") else np.nan
                except:
                    info["debtToEquity"] = np.nan

                # sector/industry if available
                info["sector"] = f_resp.get("sector")
                info["industry"] = f_resp.get("industry")
        except Exception:
            # if fundamentals parsing fails, continue with what we have
            pass

        # Historical data parsing
        hist = pd.DataFrame()
        try:
            if isinstance(ts_resp, dict) and "values" in ts_resp:
                hist = pd.DataFrame(ts_resp["values"])
                # Twelve returns newest first; convert and sort ascending
                if "datetime" in hist.columns:
                    hist.rename(columns={"datetime": "Date", "close": "Close"}, inplace=True)
                elif "date" in hist.columns:
                    hist.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
                else:
                    hist = hist.rename(columns={hist.columns[0]: "Date", hist.columns[-1]: "Close"})
                hist["Date"] = pd.to_datetime(hist["Date"])
                hist["Close"] = hist["Close"].astype(float)
                hist = hist.sort_values("Date")
            else:
                hist = pd.DataFrame()
        except Exception:
            hist = pd.DataFrame()

        return info, hist

    except Exception as e:
        # Return empty to indicate failure to caller so it falls back to yfinance
        st.warning(f"Twelve Data fetch failed for {symbol}: {e}")
        return {}, pd.DataFrame()

# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=300)
def fetch_info_and_history(symbol: str):
    """Fallback using yfinance (symbol passed without .NS)"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y", auto_adjust=False)
        if not hist.empty:
            hist = hist.reset_index()
            # rename close column consistently
            if "Close" in hist.columns:
                hist.rename(columns={"Close": "Close", "Date": "Date"}, inplace=True)
            elif "close" in hist.columns:
                hist.rename(columns={"close": "Close", "Date": "Date"}, inplace=True)
        else:
            hist = pd.DataFrame()
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Stock Screener"
])

# -------------------------
# Dashboard (Watchlist)
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    else:
        col1, col2 = st.columns([4,1])
        with col1:
            st.write(f"Watchlist: {', '.join(watchlist)}")
        with col2:
            # Let user choose primary source
            use_twelve_dashboard = st.checkbox("Prefer TwelveData (if available)", value=bool(TWELVE_API_KEY))
        if st.button("🔍 Analyze Watchlist"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, hist = ({}, pd.DataFrame())
                # Attempt Twelve if preferred and key exists
                if use_twelve_dashboard and TWELVE_API_KEY:
                    info, hist = fetch_twelvedata_data(sym)
                # If Twelve failed or empty, fallback to yfinance
                if not info:
                    info, hist = fetch_info_and_history(sym)

                # Try to find LTP
                ltp = safe_get(info, "price", np.nan) or safe_get(info, "currentPrice", np.nan) \
                      or safe_get(info, "regularMarketPrice", np.nan) or np.nan
                # compute fair value & recommendation
                fv, _ = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)
                rows.append({
                    "Symbol": sym,
                    "LTP": round(ltp, 2) if isinstance(ltp, (int, float)) and not pd.isna(ltp) else None,
                    "Fair Value": fv,
                    "Underv%": rec["undervaluation"],
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"]
                })
                progress.progress(int(((i + 1) / len(watchlist)) * 100))
            df_out = pd.DataFrame(rows)
            df_sorted = df_out.sort_values(by="Score", ascending=False)
            st.dataframe(df_sorted, use_container_width=True)
            st.success("✅ Ranked by multi-factor score (Quality + Valuation)")

# -------------------------
# Single Stock Analysis
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    use_twelve = st.checkbox("Use TwelveData API", value=bool(TWELVE_API_KEY))
    symbol = st.text_input("Enter NSE stock symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = ({}, pd.DataFrame())
        source_used = "none"
        if use_twelve and TWELVE_API_KEY:
            info, hist = fetch_twelvedata_data(symbol)
            source_used = "TwelveData" if info else "TwelveData (failed)"
        # fallback
        if not info:
            info, hist = fetch_info_and_history(symbol)
            source_used = "yfinance"
        if not info:
            st.error("No data found from TwelveData or yfinance.")
        else:
            # LTP - attempt various keys
            ltp = safe_get(info, "price", np.nan) or safe_get(info, "currentPrice", np.nan) \
                  or safe_get(info, "regularMarketPrice", np.nan) or np.nan
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)

            # Display key metrics
            cols = st.columns(4)
            cols[0].metric("Current Price (LTP)", f"{round(ltp,2) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else 'N/A'}", "")
            cols[1].metric("Fair Value", f"{fv if not pd.isna(fv) else 'N/A'}", method)
            cols[2].metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            cols[3].metric("Undervaluation %", f"{rec['undervaluation'] if rec['undervaluation'] is not None else 'N/A'}", f"Source: {source_used}")

            # show info table
            st.subheader("Raw Info (partial)")
            # Convert info to readable dataframe (short)
            try:
                info_display = {k: (round(v,4) if isinstance(v,(int,float)) and not pd.isna(v) else v) for k,v in info.items()}
            except Exception:
                info_display = info
            st.json(info_display)

            # Price chart if available
            if hist is not None and not hist.empty:
                try:
                    hist_plot = hist.set_index("Date")["Close"]
                    st.line_chart(hist_plot, height=300)
                except Exception:
                    st.write("Historical chart unavailable.")
            else:
                st.info("Historical data not available for chart.")

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV with columns: symbol, buy_price, quantity  (symbol without .NS).")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf.columns = [c.lower().strip() for c in pf.columns]
            required = {"symbol", "buy_price", "quantity"}
            if not required.issubset(set(pf.columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity")
            else:
                rows = []
                progress = st.progress(0)
                for i, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    # Fetch using yfinance (reliable for current price) - or Twelve if user prefers
                    info, _ = fetch_info_and_history(sym)
                    if not info or info.get("error"):
                        # attempt Twelve as second try
                        info_td, _ = fetch_twelvedata_data(sym) if TWELVE_API_KEY else ({}, pd.DataFrame())
                        if info_td:
                            info = info_td
                    ltp = safe_get(info, "price", np.nan) or safe_get(info, "currentPrice", np.nan) or np.nan
                    current_value = (ltp * qty) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else np.nan
                    invested = buy * qty
                    pl = current_value - invested if not pd.isna(current_value) and not pd.isna(invested) else np.nan
                    pl_pct = (pl / invested) * 100 if invested else np.nan
                    rows.append({
                        "Symbol": sym,
                        "Buy Price": round(buy,2),
                        "Qty": qty,
                        "LTP": round(ltp,2) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else None,
                        "Current Value": round(current_value,2) if not pd.isna(current_value) else None,
                        "Invested": round(invested,2),
                        "P/L": round(pl,2) if not pd.isna(pl) else None,
                        "P/L %": round(pl_pct,2) if not pd.isna(pl_pct) else None
                    })
                    progress.progress(int(((i+1)/len(pf))*100))
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error reading portfolio: " + str(e))

# -------------------------
# Alerts (manual)
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    st.info("This is a manual alert sender scaffold. For production you can wire SMTP or third-party services.")
    st.write("Set undervaluation threshold to flag/watch stocks.")
    threshold = st.number_input("Undervaluation % threshold (flag if underval >= this)", value=10.0, step=1.0)
    st.write("Upload a watchlist CSV to scan and produce flagged symbols.")
    up = st.file_uploader("Watchlist CSV (one symbol per line)", type=["csv"])
    if up:
        try:
            wl = pd.read_csv(up, header=None)[0].astype(str).str.strip().str.upper().tolist()
            flagged = []
            progress = st.progress(0)
            for i, s in enumerate(wl):
                # Prefer Twelve if available
                info, _ = (fetch_twelvedata_data(s) if TWELVE_API_KEY else ({}, pd.DataFrame()))
                if not info:
                    info, _ = fetch_info_and_history(s)
                fv, _ = estimate_fair_value(info)
                ltp = safe_get(info, "price", np.nan)
                rec = rule_based_recommendation(info, fv, ltp)
                if rec["undervaluation"] is not None and rec["undervaluation"] >= threshold:
                    flagged.append({"symbol": s, "underv%": rec["undervaluation"], "rec": rec["recommendation"]})
                progress.progress(int(((i+1)/len(wl))*100))
            if flagged:
                st.success(f"Found {len(flagged)} flagged symbols")
                st.dataframe(pd.DataFrame(flagged), use_container_width=True)
            else:
                st.info("No symbols flagged.")
        except Exception as e:
            st.error("Error processing file: " + str(e))

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    st.write("Edit your watchlist (one symbol per line). Use NSE tickers (no .NS).")
    current = load_watchlist()
    new_txt = st.text_area("Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("✅ Watchlist saved.")
        else:
            st.error("Save failed: " + msg)

# -------------------------
# Stock Screener (Nifty)
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty 500 (sample run of first 100)")
    use_twelve_screener = st.checkbox("Use TwelveData API (recommended)", value=bool(TWELVE_API_KEY))
    nifty = load_nifty_stocks()
    max_scan = st.number_input("Max symbols to scan (first N from list)", min_value=10, max_value=500, value=100, step=10)
    if st.button("Run Screener"):
        if nifty.empty:
            st.warning("Nifty list could not be loaded. Screener will use uploaded list instead.")
        rows = []
        symbols_to_scan = nifty["Symbol"].tolist()[:max_scan] if not nifty.empty else []
        if not symbols_to_scan:
            st.info("No symbols available to screen from Nifty500 archive.")
        progress = st.progress(0)
        for i, sym in enumerate(symbols_to_scan):
            info, hist = ({}, pd.DataFrame())
            if use_twelve_screener and TWELVE_API_KEY:
                info, hist = fetch_twelvedata_data(sym)
            if not info:
                info, hist = fetch_info_and_history(sym)
            price = safe_get(info, "price", np.nan) or safe_get(info, "currentPrice", np.nan) or np.nan
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": sym,
                "LTP": round(price,2) if isinstance(price,(int,float)) and not pd.isna(price) else None,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
            progress.progress(int(((i + 1) / max(1, len(symbols_to_scan))) * 100))
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)
            st.success("✅ Screener completed successfully")
        else:
            st.info("Screener returned no results.")

# -------------------------
# Footer
# -------------------------
st.caption("Made by Biswanath 🔍 | Twelve Data primary, yfinance fallback")
