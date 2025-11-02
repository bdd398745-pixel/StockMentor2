import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# -----------------------------------------------------------
# 1️⃣ PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(page_title="StockMentor", page_icon="📈", layout="wide")

st.title("📈 StockMentor — Long-Term Stock Insight (India)")
st.markdown("AI-powered insights for **long-term Indian stock investments** — using your personal watchlist.")

# -----------------------------------------------------------
# 2️⃣ LOAD WATCHLIST FROM CSV
# -----------------------------------------------------------
def load_watchlist():
    try:
        return pd.read_csv("watchlist.csv", header=None)[0].tolist()
    except Exception as e:
        st.error(f"❌ Failed to load watchlist: {e}")
        return []

def save_watchlist(stocks):
    try:
        pd.DataFrame(stocks).to_csv("watchlist.csv", index=False, header=False)
        st.success("✅ Watchlist updated successfully!")
    except Exception as e:
        st.error(f"❌ Failed to save watchlist: {e}")

watchlist = load_watchlist()

# -----------------------------------------------------------
# 3️⃣ EDIT WATCHLIST (OPTIONAL)
# -----------------------------------------------------------
with st.expander("📝 Edit My Watchlist"):
    new_text = st.text_area("Enter stock symbols (one per line, use .NS for NSE stocks):", "\n".join(watchlist))
    if st.button("💾 Save Watchlist"):
        updated_list = [x.strip() for x in new_text.split("\n") if x.strip()]
        save_watchlist(updated_list)
        st.rerun()

# -----------------------------------------------------------
# 4️⃣ HELPER FUNCTIONS
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Basic details
        current_price = info.get("currentPrice", np.nan)
        target_mean = info.get("targetMeanPrice", np.nan)
        trailing_pe = info.get("trailingPE", np.nan)
        forward_pe = info.get("forwardPE", np.nan)
        book_value = info.get("bookValue", np.nan)
        eps = info.get("trailingEps", np.nan)
        peg = info.get("pegRatio", np.nan)

        # Fair value calculation (simple estimate)
        if eps and (peg and peg > 0):
            fair_value = eps * peg * 10
        elif target_mean:
            fair_value = target_mean
        elif book_value and trailing_pe:
            fair_value = book_value * trailing_pe
        else:
            fair_value = np.nan

        if fair_value and current_price:
            undervaluation = round((fair_value - current_price) / fair_value * 100, 2)
        else:
            undervaluation = np.nan

        return {
            "Symbol": ticker,
            "Price": current_price,
            "Fair Value": round(fair_value, 2) if fair_value else np.nan,
            "Undervaluation %": undervaluation,
            "PE Ratio": trailing_pe,
            "PEG Ratio": peg,
        }

    except Exception as e:
        return {"Symbol": ticker, "Error": str(e)}

# -----------------------------------------------------------
# 5️⃣ FETCH DATA FOR ALL WATCHLIST STOCKS
# -----------------------------------------------------------
st.subheader("📊 Watchlist Insights")

if st.button("🔍 Analyze Now"):
    data = []
    with st.spinner("Fetching stock data..."):
        for stock in watchlist:
            result = get_stock_data(stock)
            data.append(result)

    df = pd.DataFrame(data)

    # Clean and display
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        undervalued = df[df["Undervaluation %"] > 0].sort_values("Undervaluation %", ascending=False)
        if not undervalued.empty:
            best_stock = undervalued.iloc[0]
            st.success(f"🏆 **Best undervalued stock now:** {best_stock['Symbol']} — {best_stock['Undervaluation %']}% undervalued.")
    else:
        st.warning("No valid stock data found.")

else:
    st.info("Click **Analyze Now** to fetch the latest stock insights.")

# -----------------------------------------------------------
# 6️⃣ FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption(f"© {datetime.now().year} StockMentor — Personal Investment Assistant by Biswanath 🧠")
