# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
Data source: Screener.in (public HTML parsing, no login)
- Keeps the same 6 tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- RJ-style scoring and ranking preserved

How to use:
- If your watchlist is on GitHub as a raw file (one symbol per line, WITHOUT .NS), set WATCHLIST_RAW_URL to that raw URL.
  Example: https://raw.githubusercontent.com/<user>/<repo>/main/watchlist.txt
- Or leave WATCHLIST_RAW_URL = None to use local watchlist.csv in the app folder.

Notes:
- Screener HTML structure may change; this parser attempts to be robust but may require small tweaks later.
- Respect Screener: this scrapes public pages and caches results to avoid aggressive traffic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import math
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="StockMentor (Screener.in)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (Screener.in)")
st.caption("Data source: Screener.in (HTML parsing). Keep scraping gentle and cache results.")

# Replace with your raw GitHub watchlist URL (one symbol per line, e.g. TCS) or None to use local file 'watchlist.csv'
WATCHLIST_RAW_URL = None
WATCHLIST_LOCAL_FILE = "watchlist.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
REQUEST_TIMEOUT = 15
CACHE_TTL = 600  # seconds
MOCK_SLEEP = 0.02
DEFAULT_PE_TARGET = 20.0

# -------------------------
# Utilities: Watchlist loader
# -------------------------
@st.cache_data(ttl=300)
def load_watchlist_from_github(raw_url: str):
    try:
        r = requests.get(raw_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        lines = [l.strip().upper() for l in r.text.splitlines() if l.strip()]
        # remove .NS if present
        lines = [l.replace('.NS','').strip() for l in lines]
        return lines
    except Exception as e:
        st.warning(f"Could not load watchlist from GitHub: {e}")
        return []

@st.cache_data(ttl=300)
def load_watchlist_local(path: str):
    try:
        df = pd.read_csv(path, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        symbols = [s.replace('.NS','').strip().upper() for s in symbols if s and str(s).strip()]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.warning(f"Error reading local watchlist: {e}")
        return []

@st.cache_data(ttl=300)
def load_watchlist():
    if WATCHLIST_RAW_URL:
        return load_watchlist_from_github(WATCHLIST_RAW_URL)
    else:
        return load_watchlist_local(WATCHLIST_LOCAL_FILE)

# -------------------------
# Screener.in HTML fetch + parse helpers
# -------------------------
def screener_company_url(symbol: str, consolidated=True):
    """Return Screener company URL for a given symbol. We assume symbol is NSE ticker form used on Screener.
    Screener uses 'company/<SYMBOL>/' paths, but sometimes the URL uses company name or id. We'll try both symbol and symbol + '/consolidated/'.
    """
    base = f"https://www.screener.in/company/{symbol}/"
    if consolidated:
        return base + "consolidated/"
    return base

@st.cache_data(ttl=CACHE_TTL)
def fetch_screener_page(symbol: str, consolidated=True):
    url = screener_company_url(symbol, consolidated=consolidated)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        # try without consolidated suffix
        if consolidated:
            try:
                url2 = screener_company_url(symbol, consolidated=False)
                r = requests.get(url2, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.text
            except Exception:
                return None
        return None


def parse_ratios_and_summary(html: str):
    """Parse main ratios like ROE, debt/equity, dividend yield, promoter holding from Screener HTML.
    Returns dict of values (float or None) and additional text where helpful.
    """
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    # 1) Try to find 'Key Ratios' table (class 'data-table' or 'snapshot')
    try:
        # Screener shows a 'key-values' div with small stats; also a 'snapshot' table
        # We'll search for text labels and adjacent values.
        text = soup.get_text(separator='|')

        # helper to find label followed by number in the HTML
        def find_label_value(label):
            # search for label in soup and get next sibling text
            el = soup.find(text=lambda t: t and label.lower() in t.lower())
            if not el:
                return None
            # try to get nearby number
            parent = el.parent
            # look next siblings and parents for numeric
            search_nodes = [parent] + list(parent.parents) + list(parent.next_siblings)[:6]
            for node in search_nodes:
                if not node:
                    continue
                txt = node.get_text(separator=' ').strip()
                # find first number like 12.34% or 123.45
                import re
                m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s?%?", txt.replace(',', ''))
                if m:
                    try:
                        val = float(m.group(1))
                        return val
                    except:
                        continue
            return None

        # common labels
        out['roe_pct'] = find_label_value('Return on Equity') or find_label_value('ROE')
        out['debt_to_equity'] = find_label_value('Debt to Equity') or find_label_value('Debt/Equity')
        out['dividend_yield_pct'] = find_label_value('Dividend Yield') or find_label_value('Dividend yield')
        out['promoter_holding_pct'] = find_label_value('Promoters') or find_label_value('Promoter Holding')

        # some values on Screener are shown as 0.00 or as ratios; promoter holding may appear like 'Promoters 50.00%'
        # convert promoter to percent if found as fraction
        for k in ['roe_pct','dividend_yield_pct','promoter_holding_pct']:
            if out.get(k) is not None and out[k] <= 3:  # likely fraction like 0.35
                out[k] = round(out[k] * 100, 2)

    except Exception:
        pass

    # 2) Extract a 'snapshot' table if present (labels on left, values on right)
    try:
        snapshot = soup.find('section', {'id': 'snapshot'})
        if snapshot:
            # find all dt/dd pairs or tr/td pairs
            # dt/dd
            dts = snapshot.find_all('dt')
            dds = snapshot.find_all('dd')
            if dts and dds and len(dts) == len(dds):
                for dt, dd in zip(dts, dds):
                    key = dt.get_text(strip=True).lower()
                    val = dd.get_text(strip=True).replace(',', '')
                    try:
                        if '%' in val:
                            out[key] = float(val.replace('%',''))
                        else:
                            out[key] = float(val)
                    except:
                        out[key] = val
    except Exception:
        pass

    return out

@st.cache_data(ttl=CACHE_TTL)
def parse_financials_for_cagr(html: str, years=3):
    """Attempt to parse the Profit & Loss / Financials table from the Screener page and compute revenue & profit CAGR over `years`.
    Returns revenue_cagr_pct, profit_cagr_pct and a DataFrame of extracted annual financials (descending recent -> older)
    """
    if not html:
        return None, None, None
    soup = BeautifulSoup(html, 'html.parser')

    # Screener often includes a table with class 'data-table' under 'Profit & Loss' or 'Consolidated Profit & Loss'.
    # We'll search for table that has 'Total Revenue' or 'Sales' or 'Net Sales' in header/index.
    tables = soup.find_all('table')
    candidate = None
    for t in tables:
        txt = t.get_text(separator='|').lower()
        if 'total revenue' in txt or 'net sales' in txt or 'net income' in txt or 'profit for the year' in txt:
            candidate = t
            break
    if candidate is None:
        # fallback: try first big table
        candidate = tables[0] if tables else None

    if candidate is None:
        return None, None, None

    # parse table into DataFrame
    try:
        df = pd.read_html(str(candidate))[0]
        # the table often has first column as metric and subsequent columns as years (recent first)
        if df.shape[1] < 2:
            return None, None, df
        # ensure first col is metric
        df = df.fillna(0)
        df.columns = [str(c) for c in df.columns]
        # If first column name is numeric (a year), it's maybe transposed format; try transpose
        first_col_name = df.columns[0]
        if any(str(c).lower().startswith('total') for c in df[first_col_name].astype(str)):
            # good
            pass
        else:
            # try transpose
            df = df.set_index(df.columns[0]).T

        # standardize index/columns
        # find revenue row
        revenue_row = None
        profit_row = None
        for r in df.index:
            rn = str(r).lower()
            if 'revenue' in rn or 'total sales' in rn or 'net sales' in rn:
                revenue_row = r
            if 'net profit' in rn or 'net income' in rn or 'profit for the year' in rn:
                profit_row = r
        # if not found, try to locate by containing 'sales' or 'income'
        if revenue_row is None:
            for r in df.index:
                if 'sale' in str(r).lower():
                    revenue_row = r; break
        if profit_row is None:
            for r in df.index:
                if 'income' in str(r).lower() or 'profit' in str(r).lower():
                    profit_row = r; break

        # convert columns to numeric years (most recent first) if possible
        # attempt to use last (years) columns
        values = None
        revenue_cagr = None
        profit_cagr = None
        if revenue_row is not None and profit_row is not None:
            rev_series = pd.to_numeric(df.loc[revenue_row].astype(str).str.replace(',','').str.replace('(','-').str.replace(')',''), errors='coerce')
            prof_series = pd.to_numeric(df.loc[profit_row].astype(str).str.replace(',','').str.replace('(','-').str.replace(')',''), errors='coerce')
            # need at least years+1 data points
            if len(rev_series.dropna()) >= years+1:
                latest = rev_series.iloc[0]
                oldest = rev_series.iloc[years]
                if pd.notna(latest) and pd.notna(oldest) and oldest != 0:
                    revenue_cagr = ((float(latest)/float(oldest))**(1.0/years)-1.0)*100.0
                    revenue_cagr = round(revenue_cagr,2)
            if len(prof_series.dropna()) >= years+1:
                latest = prof_series.iloc[0]
                oldest = prof_series.iloc[years]
                if pd.notna(latest) and pd.notna(oldest) and oldest != 0:
                    profit_cagr = ((float(latest)/float(oldest))**(1.0/years)-1.0)*100.0
                    profit_cagr = round(profit_cagr,2)
        return revenue_cagr, profit_cagr, df
    except Exception:
        return None, None, None

# -------------------------
# Fair Value Estimation (simple EPS x PE approximation using Screener numbers when possible)
# -------------------------

def estimate_fair_value_from_screener(parsed_summary: dict, info_extras: dict = None):
    """Simple fair value estimation using available EPS and P/E info from Screener page summary.
    parsed_summary: dictionary from parse_ratios_and_summary
    info_extras: optional dict (e.g., {'eps':..., 'pe':...})
    Returns (fair_value, method)
    """
    # priorities: if parsed_summary has 'eps' and 'pe' or info_extras provided, use that
    try:
        eps = None
        pe = None
        if info_extras and info_extras.get('eps'):
            eps = info_extras.get('eps')
        if info_extras and info_extras.get('pe'):
            pe = info_extras.get('pe')

        # try to extract trailing EPS from parsed_summary keys
        for k in parsed_summary.keys():
            if 'eps' in str(k).lower() or 'earnings per share' in str(k).lower():
                try:
                    eps = float(parsed_summary[k])
                except:
                    pass

        # try P/E
        for k in parsed_summary.keys():
            if 'pe' in str(k).lower() or 'p/e' in str(k).lower() or 'price to earnings' in str(k).lower():
                try:
                    pe = float(parsed_summary[k])
                except:
                    pass

        if eps is not None and pe is not None and pe>0:
            fv = eps * pe
            return round(fv,2), f"EPSxPE({pe})"
    except Exception:
        pass
    return None, 'InsufficientData'

# -------------------------
# Rule-based recommendation and RJ Score (kept same as original logic)
# -------------------------

def rule_based_recommendation_from_screener(parsed_summary: dict, fv, current_price=None, revenue_cagr_3y=None, profit_cagr_3y=None):
    # Map parsed keys to expected metrics
    try:
        roe = parsed_summary.get('roe_pct') if parsed_summary.get('roe_pct') is not None else parsed_summary.get('return on equity')
        de = parsed_summary.get('debt_to_equity') if parsed_summary.get('debt_to_equity') is not None else parsed_summary.get('debt to equity')
        cur_ratio = parsed_summary.get('current ratio') if parsed_summary.get('current ratio') is not None else parsed_summary.get('current ratio')
        pe = parsed_summary.get('pe') if parsed_summary.get('pe') is not None else parsed_summary.get('p/e')
        peg = parsed_summary.get('peg') if parsed_summary.get('peg') is not None else None
        net_margin = parsed_summary.get('net profit margin') if parsed_summary.get('net profit margin') is not None else parsed_summary.get('profit margin')
        beta = parsed_summary.get('beta') if parsed_summary.get('beta') is not None else None
        market_cap = None

        # compute undervaluation
        underv = None
        if fv and current_price and fv>0:
            try:
                underv = round(((fv - current_price) / fv) * 100, 2)
            except:
                underv = None

        score = 0
        reasons = []

        # Fundamentals
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

        # Profitability
        if isinstance(roe, (int, float)):
            if roe > 18:
                score += 10; reasons.append("Strong ROE (>18%)")
            elif roe > 12:
                score += 5; reasons.append("Good ROE (12–18%)")

        if isinstance(net_margin, (int, float)):
            if net_margin > 15:
                score += 10; reasons.append("High Profit Margin (>15%)")
            elif net_margin > 8:
                score += 5; reasons.append("Moderate Profit Margin")

        # Growth
        if isinstance(revenue_cagr_3y, (int, float)):
            if revenue_cagr_3y > 10:
                score += 10; reasons.append("Strong Sales Growth (3Y CAGR >10%)")
            elif revenue_cagr_3y > 5:
                score += 5; reasons.append("Moderate Sales Growth (3Y CAGR)")
        else:
            sales_growth = parsed_summary.get('sales growth') or parsed_summary.get('revenue growth')
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
            eps_growth = parsed_summary.get('eps growth') or parsed_summary.get('earnings growth')
            if isinstance(eps_growth, (int, float)) and eps_growth > 0.10:
                score += 10; reasons.append("Strong EPS Growth (single-year)")
            elif isinstance(eps_growth, (int, float)) and eps_growth > 0.05:
                score += 5; reasons.append("Moderate EPS Growth (single-year)")

        # Valuation
        if isinstance(pe, (int, float)) and pe > 0:
            if pe < 20:
                score += 10; reasons.append("Attractive P/E (<20)")
            elif pe < 30:
                score += 5; reasons.append("Fair P/E (<30)")

        if isinstance(peg, (int, float)) and peg < 1.5:
            score += 5; reasons.append("Reasonable PEG (<1.5)")

        # Momentum
        if isinstance(underv, (int, float)):
            if underv >= 25:
                score += 10; reasons.append("Deep undervaluation (>25%)")
            elif underv >= 10:
                score += 5; reasons.append("Undervalued (>10%)")

        # Safety
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

        return {"score": final_score, "reasons": reasons, "undervaluation_%": underv, "recommendation": rec, "market_cap": market_cap}
    except Exception:
        return {"score": 0, "reasons": [], "undervaluation_%": None, "recommendation": "Hold", "market_cap": None}

# -------------------------
# RJ Score (same logic)
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
# Email sender (same as original)
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
# Streamlit UI Tabs
# -------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor", "🏆 RJ Score"
])

# -------------------------
# Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor or set WATCHLIST_RAW_URL at top.")
    elif st.button("🔍 Analyze Watchlist"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            html = fetch_screener_page(sym, consolidated=True)
            parsed = parse_ratios_and_summary(html) if html else {}
            rev_cagr, prof_cagr, fin_df = parse_financials_for_cagr(html, years=3)
            fv, method = estimate_fair_value_from_screener(parsed, info_extras=None)
            # we don't have LTP reliably on Screener page; try to extract from parsed_summary or leave None
            ltp = None
            # try to find 'current price' text
            try:
                if html:
                    s = BeautifulSoup(html, 'html.parser')
                    cp = s.find(text=lambda t: t and 'current price' in t.lower())
                    if cp:
                        # attempt to extract nearby number
                        parent = cp.parent
                        txt = parent.get_text(separator=' ').replace(',','')
                        import re
                        m = re.search(r"([0-9]+\.?[0-9]*)", txt)
                        if m:
                            ltp = float(m.group(1))
            except Exception:
                ltp = None

            rec = rule_based_recommendation_from_screener(parsed, fv, current_price=ltp, revenue_cagr_3y=rev_cagr, profit_cagr_3y=prof_cagr)
            buy, sell = (None, None)
            try:
                if fv:
                    buy = round(fv * 0.7,2)
                    sell = round(fv * 1.2,2)
            except:
                pass

            cap = rec.get('market_cap')
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank_score = (rec.get('score',0) * 2) + (rec.get('undervaluation_%') or 0)/10 + cap_weight
            rows.append({
                "Symbol": sym,
                "LTP": ltp,
                "Fair Value": fv,
                "Underv%": rec.get('undervaluation_%'),
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec.get('recommendation'),
                "Score": rec.get('score'),
                "RankScore": round(rank_score,2),
                "Reasons": "; ".join(rec.get('reasons') or [])
            })
            progress.progress(int(((i+1)/len(watchlist))*100))
            time.sleep(MOCK_SLEEP)
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="RankScore", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Screener data)")

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("📈 Single Stock Deep Analysis (Screener.in)")
    ticker = st.text_input("Enter Stock Symbol (e.g., TCS, HDFCBANK, INFY):")
    if ticker:
        ticker = ticker.strip().upper()
        html = fetch_screener_page(ticker, consolidated=True)
        if not html:
            st.error("Could not fetch Screener page for this symbol. Check symbol or try again later.")
        else:
            parsed = parse_ratios_and_summary(html)
            rev_cagr, prof_cagr, fin_df = parse_financials_for_cagr(html, years=3)
            fv, method = estimate_fair_value_from_screener(parsed)

            # Overview
            st.subheader("📊 Overview Panel")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                cp = parsed.get('current price') or parsed.get('current-price') or None
                st.metric("Current Price", f"₹{cp:,.2f}" if isinstance(cp,(int,float)) else "-")
                st.metric("52W High", parsed.get('52 week high') or '-')
            with col2:
                st.metric("52W Low", parsed.get('52 week low') or '-')
                st.metric("Market Cap", parsed.get('market cap') or '-')
            with col3:
                pe = parsed.get('pe') or parsed.get('p/e') or '-'
                st.metric("P/E", pe)
                st.metric("P/B", parsed.get('price to book') or '-')
            with col4:
                dy = parsed.get('dividend_yield_pct') or parsed.get('dividend yield') or 0
                st.metric("Dividend Yield", f"{dy:.2f}%" if isinstance(dy,(int,float)) else '-')
                prom = parsed.get('promoter_holding_pct') or parsed.get('promoters') or '-'
                st.metric("Promoter Holding", f"{prom:.2f}%" if isinstance(prom,(int,float)) else '-')

            st.markdown("---")
            st.subheader("📈 Financial Strength")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROE", f"{parsed.get('roe_pct'):.2f}%" if isinstance(parsed.get('roe_pct'),(int,float)) else '-')
            with col2:
                st.metric("Debt/Equity", parsed.get('debt_to_equity') or '-')
            with col3:
                st.metric("Dividend Yield", f"{parsed.get('dividend_yield_pct'):.2f}%" if isinstance(parsed.get('dividend_yield_pct'),(int,float)) else '-')
            with col4:
                st.metric("Promoter%", f"{parsed.get('promoter_holding_pct'):.2f}%" if isinstance(parsed.get('promoter_holding_pct'),(int,float)) else '-')

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Revenue CAGR (3Y)", f"{rev_cagr:.2f}%" if isinstance(rev_cagr,(int,float)) else '-')
            with col2:
                st.metric("Profit CAGR (3Y)", f"{prof_cagr:.2f}%" if isinstance(prof_cagr,(int,float)) else '-')
            with col3:
                st.metric("Fair Value", f"₹{fv}" if fv else '-')

            st.markdown("---")
            st.subheader("📉 Valuation Snapshot")
            st.write(f"**Estimated Fair Value:** {fv} ({method})")

            st.markdown("---")
            st.subheader("📊 Trend Charts")
            try:
                if fin_df is not None:
                    # show revenue & profit rows if present
                    st.caption("Extracted Financials (preview)")
                    st.dataframe(fin_df.head(10))
            except Exception:
                st.warning("Unable to display financial table.")

            st.markdown("---")
            st.subheader("🧠 RJ Style Interpretation")
            st.markdown(
                "> **Think like RJ:** Look for consistent growth, ROE>15%, low D/E, cash generation and management quality.\n"
            )

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
                pf.columns = pf_columns
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    html = fetch_screener_page(sym)
                    parsed = parse_ratios_and_summary(html) if html else {}
                    ltp = parsed.get('current price') or None
                    try:
                        if isinstance(ltp, str):
                            ltp = float(ltp.replace(',',''))
                    except:
                        ltp = None
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    current_value = round((ltp * qty),2) if isinstance(ltp,(int,float)) else None
                    invested = round(buy*qty,2)
                    pl = round((current_value - invested),2) if current_value is not None else None
                    pl_pct = round((pl/invested*100),2) if pl is not None and invested!=0 else None
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
# Alerts
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
                html = fetch_screener_page(sym)
                parsed = parse_ratios_and_summary(html) if html else {}
                rev_cagr, prof_cagr, _ = parse_financials_for_cagr(html, years=3)
                fv, method = estimate_fair_value_from_screener(parsed)
                ltp = parsed.get('current price') or None
                underv = None
                if fv and ltp and fv>0:
                    try:
                        underv = round(((fv - float(ltp))/fv)*100,2)
                    except:
                        underv = None
                if isinstance(underv,(int,float)) and underv >= underv_threshold:
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
    # Load current
    current = load_watchlist()
    new_txt = st.text_area("Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist locally"):
        try:
            new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
            pd.DataFrame(new_list).to_csv(WATCHLIST_LOCAL_FILE, index=False, header=False)
            st.success(f"Watchlist saved to {WATCHLIST_LOCAL_FILE}. Reload Dashboard to analyze.")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.markdown("---")
    st.write("If your watchlist is on GitHub, set WATCHLIST_RAW_URL in the script top to point to the raw file (one symbol per line).")

# -------------------------
# RJ Score Tab
# -------------------------
with tab6:
    st.header("🏆 RJ Score — Jhunjhunwala-Style Hybrid Stock Scoring System")
    st.markdown("Combines fundamentals (data) + qualitative conviction (user inputs).")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("⚠️ Watchlist empty. Add symbols in Watchlist Editor.")
    else:
        with st.expander("Scoring parameters / defaults"):
            market_phase = st.selectbox("Market Phase", ["neutral","bull","bear"], index=0)
            management_quality = st.slider("Management quality (1-5)", 1, 5, 4)
            moat_strength = st.slider("Moat strength (1-5)", 1, 5, 3)
            growth_potential = st.slider("Growth potential (1-5)", 1, 5, 4)
        if st.button("🏁 Run RJ Scoring"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                html = fetch_screener_page(sym)
                parsed = parse_ratios_and_summary(html) if html else {}
                rev_cagr, prof_cagr, _ = parse_financials_for_cagr(html, years=3)
                roe_display = parsed.get('roe_pct') or 0
                debt_eq = parsed.get('debt_to_equity') or 0
                rev_cagr_val = rev_cagr or 0
                prof_cagr_val = prof_cagr or 0
                pe_ratio = parsed.get('pe') or parsed.get('p/e') or DEFAULT_PE_TARGET
                pe_industry = DEFAULT_PE_TARGET
                div_yield = parsed.get('dividend_yield_pct') or 0
                promoter_hold = parsed.get('promoter_holding_pct') or 0

                result = stock_score(
                    roe_display or 0,
                    debt_eq or 0,
                    rev_cagr_val or 0,
                    prof_cagr_val or 0,
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
                    "D/E": round(debt_eq or 0,2),
                    "Rev CAGR%": round(rev_cagr_val,1),
                    "Profit CAGR%": round(prof_cagr_val,1),
                    "Div Yield%": round(div_yield,2),
                    "Promoter%": round(promoter_hold,1),
                    "RJ Score": result["Score"],
                    "Rating": result["Rating"]
                })
                progress.progress(int(((i+1)/len(watchlist))*100))
                time.sleep(MOCK_SLEEP)
            df = pd.DataFrame(rows)
            df_sorted = df.sort_values(by="RJ Score", ascending=False)
            st.dataframe(df_sorted, use_container_width=True)
            st.success("✅ RJ-style ranking complete — blending fundamentals with conviction!")

# Footer
st.markdown("---")
st.caption("StockMentor — Screener.in backed, rule-based long-term stock helper. Use responsibly and respect site terms.")
