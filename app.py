import streamlit as st
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI
from datetime import datetime

# === OpenAI client (new API format) ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Page Config ===
st.set_page_config(page_title="StockMentor AI", layout="wide", page_icon="📈")

st.title("📈 StockMentor – Your AI Stock Assistant")
st.caption("Powered by OpenAI GPT + Yahoo Finance")

# === Load Watchlist ===
@st.cache_data
def load_watchlist():
    try:
        return pd.read_csv("watchlist.csv")["Symbol"].tolist()
    except Exception:
        return ["RELIANCE", "HDFCBANK", "TCS"]

watchlist = load_watchlist()

# === Fetch Stock Data ===
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    data = yf.Ticker(symbol)
    info = data.info
    hist = data.history(period="6mo")
    return info, hist

# === Calculate Fair Value (DCF style simplified) ===
def fair_value(info):
    try:
        eps = info.get("trailingEps", 0)
        growth = info.get("earningsGrowth", 0.1)
        discount = 0.1
        intrinsic = eps * (1 + growth) / discount
        return round(intrinsic, 2)
    except Exception:
        return None

# === AI Insight Generator ===
def ai_analysis(symbol, info):
    try:
        prompt = f"""
        Analyze {symbol} stock using the following data:
        Company Name: {info.get('longName')}
        Sector: {info.get('sector')}
        Current Price: {info.get('currentPrice')}
        Market Cap: {info.get('marketCap')}
        Forward PE: {info.get('forwardPE')}
        Beta: {info.get('beta')}
        Return a short summary with 3 points:
        1. Investment Outlook
        2. Risk Factors
        3. Recommendation (Buy/Hold/Sell)
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {e}"

# === Email Alert ===
def send_email_alert(symbol, price, target, recipient):
    try:
        msg = MIMEText(f"{symbol} has reached ₹{price}, crossing your target ₹{target}.")
        msg["Subject"] = f"📢 Stock Alert: {symbol}"
        msg["From"] = st.secrets["EMAIL_SENDER"]
        msg["To"] = recipient

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(st.secrets["EMAIL_SENDER"], st.secrets["EMAIL_PASSWORD"])
            server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email failed: {e}")
        return False

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 AI Insights", "🔔 Alerts", "⚙️ Settings"])

# === TAB 1: Dashboard ===
with tab1:
    st.header("Live Watchlist Data")
    selected = st.selectbox("Select Stock", watchlist)
    info, hist = get_stock_data(selected)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{info.get('currentPrice', 0)}")
    col2.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0)}")
    col3.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0)}")

    fv = fair_value(info)
    if fv:
        st.info(f"**Estimated Fair Value:** ₹{fv}")

    st.line_chart(hist["Close"], use_container_width=True)

# === TAB 2: AI Insights ===
with tab2:
    st.header("AI-Powered Analysis")
    symbol = st.selectbox("Choose stock for AI analysis", watchlist, key="ai_select")
    info, _ = get_stock_data(symbol)
    if st.button("Generate AI Insight"):
        with st.spinner("Analyzing..."):
            st.write(ai_analysis(symbol, info))

# === TAB 3: Alerts ===
with tab3:
    st.header("Price Alert Notifications")
    alert_symbol = st.selectbox("Select Stock", watchlist, key="alert_stock")
    target_price = st.number_input("Target Price (₹)", min_value=1.0, step=1.0)
    email = st.text_input("Your Email")

    if st.button("Set Alert"):
        current_price = yf.Ticker(alert_symbol).info.get("currentPrice", 0)
        if current_price >= target_price:
            if send_email_alert(alert_symbol, current_price, target_price, email):
                st.success("Alert triggered and email sent!")
        else:
            st.info(f"Current ₹{current_price} < Target ₹{target_price}. Will alert later.")

# === TAB 4: Settings ===
with tab4:
    st.header("Manage Watchlist")
    new_stock = st.text_input("Add a stock symbol (e.g., TCS, INFY)")
    if st.button("Add to Watchlist"):
        df = pd.DataFrame(watchlist, columns=["Symbol"])
        if new_stock not in df["Symbol"].values:
            df.loc[len(df)] = new_stock
            df.to_csv("watchlist.csv", index=False)
            st.success(f"Added {new_stock}!")
        else:
            st.warning("Already exists in watchlist.")
    st.dataframe(pd.DataFrame(watchlist, columns=["Symbol"]))
    st.caption("You can also directly edit `watchlist.csv` and reload the app.")
