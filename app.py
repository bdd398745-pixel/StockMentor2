# app.py
"""
StockMentor (Enhanced)
Single-file Streamlit app with:
 - Watchlist read/save (watchlist.csv)
 - Fair value estimation (targetMeanPrice OR PE * EPS OR bookValue * PE)
 - Multi-tab UI: Dashboard, Stock Insights, AI Mentor, Alerts, Portfolio, Watchlist Editor
 - Optional OpenAI integration (paste API key for richer text analysis)
 - Email alerts via SMTP (send immediate alerts)
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.message import EmailMessage
from datetime import datetime
import time
import math

# Optional: if you want to use OpenAI in AI Mentor tab, install openai and uncomment below
# import openai

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="StockMentor (Enhanced)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Enhanced Long-Term Advisor (India)")
st.caption("Personal app: long-term Indian stocks. No paid APIs required. Optional OpenAI for richer text.")

# ---------------------- UTILITIES: WATCHLIST ----------------------
WATCHLIST_FILE = "watchlist.csv"

def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        # Remove empty lines
        symbols = [s for s in symbols if s]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error("Error loading watchlist.csv: " + str(e))
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# ---------------------- DATA FETCH & VALUATION ----------------------
@st.cache_data(ttl=1800)
def fetch_yf_info(symbol):
    """Fetch info dict and 5y history from yfinance for the given .NS symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="5y")  # for charts
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

def estimate_fair_value_from_info(info):
    """
    Compute a conservative fair value:
    Priority:
     1) info['targetMeanPrice'] (analyst consensus) if present & reasonable
     2) trailingEps * fallback_pe
     3) bookValue * trailingPE (if bookValue present)
    fallback_pe chosen from info['forwardPE']/['trailingPE'] else 20 as default
    """
    try:
        # Try analyst target
        target = info.get("targetMeanPrice")
        current_price = info.get("currentPrice")
        trailing_pe = info.get("trailingPE") or info.get("trailingPE", None)
        forward_pe = info.get("forwardPE")
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")

        # Heuristics to pick pe target
        pe_target = None
        if forward_pe and forward_pe > 0 and forward_pe < 200:
            pe_target = forward_pe
        elif trailing_pe and trailing_pe > 0 and trailing_pe < 200:
            # pick a slightly lower to be conservative
            pe_target = max(10, trailing_pe * 0.9)
        else:
            pe_target = 20  # default conservative

        # 1) use targetMeanPrice if numeric and > 0
        if target and isinstance(target, (int, float)) and not math.isnan(target) and target > 0:
            return round(float(target), 2), "AnalystTarget"

        # 2) use EPS * pe_target
        if eps and pe_target:
            # if eps is negative, skip
            if eps > 0:
                fv = eps * pe_target
                return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"

        # 3) fallback book value * pe (less ideal but a fallback)
        if book_value and trailing_pe:
            fv = book_value * trailing_pe
            return round(float(fv), 2), "BVxPE"

        # If nothing computable
        return None, "InsufficientData"
    except Exception as e:
        return None, f"Error:{e}"

# ---------------------- EMAIL ALERTS ----------------------
def send_email(smtp_server, smtp_port, username, password, sender, recipients, subject, body):
    """Send an email. recipients: list or comma-separated string"""
    try:
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(",") if r.strip()]
        msg = EmailMessage()
        msg["From"] = sender or username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        # Connect and send (supports TLS)
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=20)
        server.ehlo()
        if smtp_port in (587, 25):
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# ---------------------- AI MENTOR (RULE-BASED & OPTIONAL OPENAI) ----------------------
def rule_based_opinion(info, fair_value, undervaluation):
    """
    Simple rule-based opinion:
     - Strong Buy if undervaluation >= 15% and ROE>15 and D/E reasonable
     - Buy if undervaluation between 7-15 and ROE positive
     - Hold if near fair value (-7% to +7%)
     - Avoid if overvalued (undervaluation < -7) or negative fundamentals
    """
    try:
        pe = info.get("trailingPE") or 0
        roe = info.get("returnOnEquity") or 0
        de = info.get("debtToEquity") or 999
        eps = info.get("trailingEps") or 0
        if undervaluation is None or (isinstance(undervaluation, float) and np.isnan(undervaluation)):
            return "No fair value computed — insufficient data."

        if undervaluation >= 15 and roe and roe > 0.12 and (de < 2 or de == 0):
            return "💚 Strong Buy — materially undervalued with good ROE & manageable leverage."
        if undervaluation >= 7 and roe and roe > 0.08:
            return "🟢 Buy — undervalued with reasonable fundamentals."
        if -7 <= undervaluation < 7:
            return "🟡 Hold — near fair value; hold and monitor fundamentals."
        if undervaluation < -7:
            # overpriced
            if eps < 0:
                return "🔴 Avoid — company showing negative EPS and currently overvalued."
            return "🔴 Overvalued — not attractive at current price."
        return "Hold — insufficient specific signals."
    except Exception as e:
        return "Error in rule-based opinion: " + str(e)

# Optional: function to call OpenAI if user provides key (uncomment openai import & install openai lib)
def openai_opinion(api_key, prompt):
    """
    Example usage: requires openai package. This function is illustrative.
    If you want full natural language analysis, install 'openai' and uncomment import line above.
    """
    try:
        import openai
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or gpt-4o if available; change as you like
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise stock investment reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ---------------------- APP UI: Tabs ----------------------
tabs = st.tabs(["🏠 Dashboard", "📊 Stock Insights", "🧠 AI Mentor", "📣 Alerts (Email)", "💼 Portfolio", "🧾 Watchlist Editor"])

# ---------------------- TAB: DASHBOARD ----------------------
with tabs[0]:
    st.header("🏠 Dashboard — Quick Summary")
    watchlist = load_watchlist()
    st.write(f"Watchlist loaded: {len(watchlist)} symbols")
    if len(watchlist) == 0:
        st.info("Your watchlist.csv is empty. Go to 'Watchlist Editor' tab to add symbols (use .NS for NSE).")
    else:
        if st.button("🔍 Analyze watchlist now"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, hist = fetch_yf_info(sym)
                # If info has 'error' key, handle
                if info.get("error"):
                    rows.append({"Symbol": sym, "Price": None, "Fair Value": None, "Undervaluation %": None, "Status": f"Error:{info.get('error')}"})
                else:
                    current_price = info.get("currentPrice") or np.nan
                    fv, method = estimate_fair_value_from_info(info)
                    if fv:
                        underv = round((fv - (current_price or 0)) / fv * 100, 2) if current_price and fv else None
                    else:
                        underv = None
                    rows.append({
                        "Symbol": sym,
                        "Price": current_price,
                        "Fair Value": fv,
                        "Undervaluation %": underv,
                        "ValuationMethod": method,
                        "PE": info.get("trailingPE"),
                        "ROE": info.get("returnOnEquity"),
                        "D/E": info.get("debtToEquity")
                    })
                progress.progress(int(((i+1)/len(watchlist))*100))
                time.sleep(0.05)  # small pause improves progress UX

            df = pd.DataFrame(rows)
            st.dataframe(df.sort_values(by="Undervaluation %", ascending=False, na_position="last"), use_container_width=True)
            # Top picks
            df_valid = df[df["Undervaluation %"].notna()]
            if not df_valid.empty:
                top = df_valid.sort_values("Undervaluation %", ascending=False).head(3)
                st.markdown("### 🏆 Top undervalued picks (by % undervaluation)")
                for _, r in top.iterrows():
                    st.success(f"{r['Symbol']} — {r['Undervaluation %']}% undervalued (Method: {r['ValuationMethod']})")
            else:
                st.info("No computed fair values for your watchlist (insufficient data).")

# ---------------------- TAB: STOCK INSIGHTS ----------------------
with tabs[1]:
    st.header("📊 Stock Insights (single)")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Add symbols in Watchlist Editor first.")
    else:
        sel = st.selectbox("Select symbol", watchlist)
        if sel:
            info, hist = fetch_yf_info(sel)
            if info.get("error"):
                st.error("Data error: " + info.get("error"))
            else:
                cur = info.get("currentPrice")
                fv, method = estimate_fair_value_from_info(info)
                underv = None
                if fv and cur:
                    underv = round((fv - cur) / fv * 100, 2)
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{cur}" if cur else "-")
                col2.metric("Fair Value", f"₹{fv}" if fv else "-")
                col3.metric("Undervaluation %", f"{underv}%" if underv is not None else "-")
                st.write("**Key fundamentals:**")
                st.write({
                    "PE": info.get("trailingPE"),
                    "Forward PE": info.get("forwardPE"),
                    "EPS (TTM)": info.get("trailingEps"),
                    "ROE": info.get("returnOnEquity"),
                    "Debt/Equity": info.get("debtToEquity"),
                    "Book Value": info.get("bookValue")
                })
                st.markdown("---")
                st.write("📈 Price history (5 years)")
                if hist is not None and not hist.empty:
                    st.line_chart(hist["Close"])
                else:
                    st.info("No historical price data available.")

# ---------------------- TAB: AI MENTOR ----------------------
with tabs[2]:
    st.header("🧠 AI Mentor — Buy / Hold / Sell (Rule-based + Optional OpenAI)")
    st.markdown("This tab gives a quick rule-based opinion. For richer natural language, paste your OpenAI API key below (optional).")
    api_key = st.text_input("Optional: OpenAI API key (paste to enable richer text output)", type="password")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Add symbols in Watchlist Editor first.")
    else:
        run_ai = st.button("🔎 Generate AI Opinions for Watchlist")
        if run_ai:
            results = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, _ = fetch_yf_info(sym)
                if info.get("error"):
                    results.append({"Symbol": sym, "Opinion": "Data error"})
                else:
                    fv, method = estimate_fair_value_from_info(info)
                    cur = info.get("currentPrice") or np.nan
                    underv = None
                    if fv and cur and cur>0:
                        underv = round((fv - cur) / fv * 100, 2)
                    rule_op = rule_based_opinion(info, fv, underv)
                    if api_key:
                        # build a prompt
                        prompt = (
                            f"Provide a concise buy/hold/sell opinion for the Indian NSE stock {sym}.\n"
                            f"Current price: {cur}\nEstimated fair value: {fv} (method: {method})\n"
                            f"Undervaluation%: {underv}\nKey fundamentals:\n"
                            f"PE: {info.get('trailingPE')}\nROE: {info.get('returnOnEquity')}\nDebtToEquity: {info.get('debtToEquity')}\nKeep it short (2-4 lines) and give 1 short reason and 1 risk."
                        )
                        ai_text = openai_opinion(api_key, prompt)
                        opinion = ai_text
                    else:
                        opinion = rule_op
                    results.append({"Symbol": sym, "Opinion": opinion, "Underv%": underv, "Method": method})
                progress.progress(int(((i+1)/len(watchlist))*100))
                time.sleep(0.02)
            df_op = pd.DataFrame(results)
            st.dataframe(df_op, use_container_width=True)

# ---------------------- TAB: ALERTS ----------------------
with tabs[3]:
    st.header("📣 Alerts (Email)")
    st.markdown("Configure SMTP and send immediate alerts. For Gmail: create an *App Password* and use smtp.gmail.com with port 587.")
    with st.form("email_form"):
        smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", value=587)
        smtp_user = st.text_input("SMTP username (email)")
        smtp_pass = st.text_input("SMTP password (app password recommended)", type="password")
        sender = st.text_input("From (optional)", value=smtp_user)
        recipients = st.text_input("Recipients (comma separated emails)")
        subject = st.text_input("Subject", value="StockMentor Alert")
        send_threshold = st.number_input("Send alerts for undervaluation % >= ", value=10)
        submit_email = st.form_submit_button("📨 Send alerts now")

    if submit_email:
        wl = load_watchlist()
        if not wl:
            st.error("Watchlist empty.")
        elif not smtp_user or not smtp_pass or not recipients:
            st.error("Provide SMTP username/password and recipient(s).")
        else:
            # Build alerts body
            lines = []
            any_alert = False
            for s in wl:
                info, _ = fetch_yf_info(s)
                if info.get("error"):
                    continue
                fv, method = estimate_fair_value_from_info(info)
                cur = info.get("currentPrice") or np.nan
                if fv and cur and fv>0:
                    underv = round((fv - cur) / fv * 100, 2)
                else:
                    underv = None
                if underv is not None and underv >= send_threshold:
                    any_alert = True
                    lines.append(f"{s}: Current ₹{cur} | Fair ₹{fv} ({method}) | Undervalued {underv}%")
            if not any_alert:
                st.info("No stocks passed the threshold for alerts.")
            else:
                body = "StockMentor alerts:\n\n" + "\n".join(lines) + "\n\nGenerated by StockMentor."
                ok, msg = send_email(smtp_server, int(smtp_port), smtp_user, smtp_pass, sender, recipients, subject, body)
                if ok:
                    st.success("Alerts sent successfully.")
                else:
                    st.error("Failed to send email: " + msg)

# ---------------------- TAB: PORTFOLIO ----------------------
with tabs[4]:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload a CSV with columns: symbol (with .NS), buy_price, quantity")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            # validate
            if not set(["symbol", "buy_price", "quantity"]).issubset(set(pf.columns.str.lower())):
                st.error("CSV must have columns: symbol, buy_price, quantity (case-insensitive).")
            else:
                # normalize column names
                pf.columns = pf.columns.str.lower()
                pf["symbol"] = pf["symbol"].str.strip()
                rows = []
                for _, row in pf.iterrows():
                    s = row["symbol"]
                    buy = float(row["buy_price"])
                    qty = float(row["quantity"])
                    info, _ = fetch_yf_info(s)
                    if info.get("error"):
                        cur = np.nan
                    else:
                        cur = info.get("currentPrice") or np.nan
                    current_value = cur * qty if not np.isnan(cur) else np.nan
                    invested = buy * qty
                    pl = current_value - invested if not np.isnan(current_value) else np.nan
                    plpct = (pl / invested * 100) if invested and not np.isnan(pl) else np.nan
                    rows.append({
                        "symbol": s,
                        "buy_price": buy,
                        "quantity": qty,
                        "current_price": cur,
                        "current_value": current_value,
                        "invested_value": invested,
                        "P/L": pl,
                        "P/L %": plpct
                    })
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total Realised/Unrealised P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error processing portfolio: " + str(e))

# ---------------------- TAB: WATCHLIST EDITOR ----------------------
with tabs[5]:
    st.header("🧾 Watchlist Editor")
    st.markdown("Edit your watchlist below (one symbol per line). Use the NSE symbol with `.NS` suffix (e.g., INFY.NS).")
    current = load_watchlist()
    txt = st.text_area("Watchlist (one per line)", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
        new_list = [s.strip() for s in txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("Watchlist saved. Reload or go to Dashboard to analyze.")
        else:
            st.error("Failed to save: " + msg)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption(f"StockMentor — personal long-term investing helper. Data via Yahoo Finance (yfinance). {datetime.now().year}")
