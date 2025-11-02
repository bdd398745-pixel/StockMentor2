import streamlit as st
import pandas as pd
import yfinance as yf
from openai import OpenAI
from datetime import datetime

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="📈 StockMentor", layout="wide", page_icon="💹")
st.title("💹 StockMentor – Smart Stock Analyzer")
st.caption("Live data + AI insights + valuation dashboard")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================
# LOAD WATCHLIST
# ==========================
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv("watchlist.csv")
        if "Symbol" in df.columns:
            return df["Symbol"].tolist()
        else:
            return df.iloc[:, 0].tolist()
    except Exception as e:
        st.warning(f"⚠️ Could not load watchlist.csv: {e}")
        return ["RELIANCE", "HDFCBANK", "TCS"]

watchlist = load_watchlist()

# ==========================
# FETCH STOCK DATA
# ==========================
@st.cache_data(ttl=600)
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")  # Add .NS for NSE stocks
        info = ticker.info
        price = info.get("currentPrice") or info.get("lastPrice") or None
        eps = info.get("trailingEps", 0)
        growth = info.get("earningsGrowth", 0.08)
        pe = info.get("trailingPE", 0)
        name = info.get("longName", symbol)
        return {
            "Name": name,
            "Symbol": symbol,
            "LTP": round(price, 2) if price else None,
            "EPS": eps,
            "PE": pe,
            "Growth": growth
        }
    except Exception as e:
        return {"Name": symbol, "Symbol": symbol, "LTP": None, "EPS": None, "PE": None, "Growth": None}

# ==========================
# FAIR VALUE CALCULATION
# ==========================
def calc_fair_value(eps, growth, discount=0.1):
    try:
        if eps is None or eps <= 0:
            return None
        if growth is None or growth <= 0:
            growth = 0.08
        fv = eps * (1 + growth) / (discount - growth/2)
        return round(fv, 2)
    except Exception:
        return None

# ==========================
# AI INSIGHT FUNCTION
# ==========================
def ai_summary(symbol, data):
    try:
        prompt = f"""
        You are a financial analyst. Analyze {symbol} using the following data:
        EPS: {data.get('EPS')}
        PE: {data.get('PE')}
        Growth: {data.get('Growth')}
        LTP: {data.get('LTP')}
        Fair Value (approx): {data.get('FairValue')}
        Give a short, simple summary:
        - Investment Outlook
        - Risk Level
        - Recommendation (Buy/Hold/Sell)
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI summary unavailable: {e}"

# ==========================
# MAIN TABS
# ==========================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Insights", "⚙️ Settings"])

# ==========================
# TAB 1 – DASHBOARD
# ==========================
with tab1:
    st.header("📊 Watchlist Overview")
    st.caption("Live market data + fair value comparison")

    data_list = []
    for symbol in watchlist:
        d = get_stock_data(symbol)
        d["FairValue"] = calc_fair_value(d["EPS"], d["Growth"])
        if d["LTP"] and d["FairValue"]:
            d["Valuation Ratio"] = round(d["LTP"] / d["FairValue"], 2)
        else:
            d["Valuation Ratio"] = None
        data_list.append(d)

    df = pd.DataFrame(data_list)

    # Color coding for valuation ratio
    def color_ratio(val):
        if val is None:
            return ""
        if val < 0.9:
            return "background-color:#b7f7b7"  # Undervalued
        elif val <= 1.1:
            return "background-color:#fff6b3"  # Fairly valued
        else:
            return "background-color:#ffb3b3"  # Overvalued

    st.dataframe(
        df.style.applymap(color_ratio, subset=["Valuation Ratio"]),
        use_container_width=True,
    )

    st.caption("🟢 <0.9 = Undervalued | 🟡 0.9–1.1 = Fair | 🔴 >1.1 = Overvalued")

# ==========================
# TAB 2 – AI INSIGHTS
# ==========================
with tab2:
    st.header("🤖 AI-Powered Stock Analysis")
    selected = st.selectbox("Select a stock for AI insights", watchlist)
    data = [d for d in data_list if d["Symbol"] == selected][0]
    if st.button("Generate AI Insight"):
        with st.spinner("Analyzing via GPT..."):
            insight = ai_summary(selected, data)
            st.write(insight)

# ==========================
# TAB 3 – SETTINGS
# ==========================
with tab3:
    st.header("⚙️ Manage Watchlist")
    st.caption("You can add new stock symbols below or manually edit watchlist.csv")

    new_symbol = st.text_input("Add stock symbol (e.g., TCS, HDFCBANK)")
    if st.button("Add to watchlist"):
        if new_symbol and new_symbol not in watchlist:
            watchlist.append(new_symbol)
            pd.DataFrame({"Symbol": watchlist}).to_csv("watchlist.csv", index=False)
            st.success(f"{new_symbol} added successfully!")
        else:
            st.warning("Symbol already exists or is invalid.")

    st.dataframe(pd.DataFrame({"Symbol": watchlist}))
