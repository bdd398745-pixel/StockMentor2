# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt

st.set_page_config(page_title="StockMentor – Long-Term Stock Advisor (India)", page_icon="📈", layout="wide")

# ---------------------- HEADER ----------------------
st.title("📈 StockMentor – Long-Term Stock Advisor (India)")
st.markdown("Your personal AI-powered long-term stock analysis app 🇮🇳")

# ---------------------- LOAD WATCHLIST ----------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv("watchlist.csv")
        return df
    except Exception as e:
        st.error("⚠️ Could not load watchlist.csv. Please ensure it’s uploaded.")
        return pd.DataFrame(columns=["symbol"])

watchlist = load_watchlist()
if watchlist.empty:
    st.stop()

# ---------------------- FETCH FUNDAMENTAL DATA ----------------------
@st.cache_data
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        return {
            "symbol": symbol,
            "current_price": info.get("currentPrice"),
            "fair_value": info.get("targetMeanPrice", None),
            "pe_ratio": info.get("trailingPE", None),
            "roe": info.get("returnOnEquity", None),
            "de_ratio": info.get("debtToEquity", None)
        }
    except:
        return None

def calc_undervaluation(row):
    if row["fair_value"] and row["current_price"]:
        return round(((row["fair_value"] - row["current_price"]) / row["fair_value"]) * 100, 1)
    return None

# ---------------------- BUILD DATAFRAME ----------------------
data_list = []
for s in watchlist["symbol"]:
    d = get_stock_data(s)
    if d:
        data_list.append(d)

df = pd.DataFrame(data_list)
df["undervaluation_%"] = df.apply(calc_undervaluation, axis=1)

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Watchlist Overview",
    "📊 Single Stock",
    "📈 Trend Analysis",
    "💬 AI Mentor Insights",
    "💼 Portfolio Tracker"
])

# ---------------------- WATCHLIST OVERVIEW ----------------------
with tab1:
    st.header("📋 Your Watchlist Overview")
    st.dataframe(df, use_container_width=True)

    best_stock = df.loc[df["undervaluation_%"].idxmax()] if not df.empty else None
    if best_stock is not None:
        st.success(f"🏆 Best undervalued stock currently: **{best_stock.symbol}** with **{best_stock.undervaluation_%}%** undervaluation.")

# ---------------------- SINGLE STOCK ----------------------
with tab2:
    st.header("🔍 Single Stock View")
    stock = st.selectbox("Select stock to analyze", watchlist["symbol"])

    if stock:
        st.subheader(f"📊 {stock} Overview")
        info = get_stock_data(stock)
        if info:
            st.json(info)

# ---------------------- TREND ANALYSIS ----------------------
with tab3:
    st.header("📈 Price Trend Analysis")
    stock = st.selectbox("Select stock for trend chart", watchlist["symbol"], key="trend")
    period = st.selectbox("Select period", ["6mo", "1y", "2y"], key="period")
    data = yf.download(stock + ".NS", period=period)
    if not data.empty:
        st.line_chart(data["Close"])
        st.caption(f"Price trend for {stock} over last {period}")

# ---------------------- AI MENTOR INSIGHTS ----------------------
def generate_ai_opinion(row):
    if row["undervaluation_%"] and row["undervaluation_%"] > 10 and row["pe_ratio"] and row["pe_ratio"] < 25:
        return "💚 Strong Buy – undervalued and reasonably priced."
    elif row["undervaluation_%"] and -5 < row["undervaluation_%"] <= 10:
        return "🟡 Hold – near fair value."
    else:
        return "🔴 Avoid / Overvalued."

with tab4:
    st.header("💬 AI Mentor Insights")
    df["AI_Opinion"] = df.apply(generate_ai_opinion, axis=1)
    st.dataframe(df[["symbol", "undervaluation_%", "pe_ratio", "roe", "AI_Opinion"]], use_container_width=True)

# ---------------------- PORTFOLIO TRACKER ----------------------
with tab5:
    st.header("💼 Portfolio Tracker")
    st.caption("Upload your portfolio CSV with columns: symbol, buy_price, quantity")
    uploaded = st.file_uploader("Upload Portfolio", type="csv")

    if uploaded:
        pf = pd.read_csv(uploaded)
        merged = pd.merge(pf, df[["symbol", "current_price"]], on="symbol", how="left")
        merged["current_value"] = merged["current_price"] * merged["quantity"]
        merged["invested_value"] = merged["buy_price"] * merged["quantity"]
        merged["profit_loss"] = merged["current_value"] - merged["invested_value"]
        merged["profit_%"] = (merged["profit_loss"] / merged["invested_value"]) * 100
        st.dataframe(merged, use_container_width=True)
        total = merged["profit_loss"].sum()
        st.metric("💰 Total Portfolio P/L", f"{total:,.2f} ₹")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("Created by Biswanath • Powered by Streamlit + Yahoo Finance API (Free)")
