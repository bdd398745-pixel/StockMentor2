# 📈 StockMentor – Long-Term Stock Advisor (India)

Your personal **AI-powered stock advisor** built for **long-term Indian investors** 🇮🇳  
This app helps you analyze your watchlist, identify undervalued stocks, track portfolio profit/loss, and receive AI insights on when to **Buy**, **Hold**, or **Avoid** stocks.

---

## 🚀 Features

✅ **Watchlist Overview**
- View your chosen stock list with live prices and fundamental data.
- Automatically calculates **undervaluation %**, **P/E**, **ROE**, and **Debt-Equity** ratio.
- Highlights the **best undervalued stock** in your list.

✅ **Single Stock View**
- View all financial details for any one stock (INFY, TCS, RELIANCE, etc.).
- Pulls real-time data from Yahoo Finance.

✅ **Trend Analysis**
- Visualize stock price history (6M, 1Y, 2Y).
- Understand long-term momentum and price movement trends.

✅ **AI Mentor Insights**
- Generates smart opinions:
  - 💚 **Strong Buy** – undervalued and fundamentally sound  
  - 🟡 **Hold** – near fair value  
  - 🔴 **Avoid / Overvalued** – priced too high or weak fundamentals

✅ **Portfolio Tracker**
- Upload your portfolio (symbol, buy_price, quantity).
- Automatically calculates total investment, current value, and P/L%.
- Shows overall portfolio profit/loss in ₹.

✅ **Runs Free on Streamlit Cloud**
- No API keys required.
- Fetches stock data directly from Yahoo Finance (India NSE).

---

## 🧠 Tech Stack

- **Language:** Python 🐍  
- **Framework:** Streamlit 🌐  
- **Data Source:** Yahoo Finance (via `yfinance`)  
- **Libraries:**  
  - `pandas` – data processing  
  - `yfinance` – stock data  
  - `streamlit` – front-end app

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/StockMentor.git
cd StockMentor
