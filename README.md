# 📈 StockMentor — Long-Term Investment Assistant (India)

This is your personal **AI-inspired long-term investment dashboard** for Indian stocks.  
It analyzes your custom watchlist, estimates fair value, and identifies undervalued opportunities.

---

## 🚀 Features
| Factor                           | Metric                  | Weight | Logic                                               |
| -------------------------------- | ----------------------- | ------ | --------------------------------------------------- |
| 🧾 **1. Fundamentals (20 pts)**  | Debt-to-Equity          | 10     | +10 if D/E < 0.5; +5 if 0.5–1; else 0               |
|                                  | Current Ratio           | 10     | +10 if > 1.5; +5 if 1–1.5; else 0                   |
| 💰 **2. Profitability (20 pts)** | Return on Equity (ROE)  | 10     | +10 if > 18%; +5 if 12–18%; else 0                  |
|                                  | Net Profit Margin       | 10     | +10 if > 15%; +5 if 8–15%; else 0                   |
| 📈 **3. Growth (20 pts)**        | 5-Year Sales CAGR       | 10     | +10 if > 10%; +5 if 5–10%; else 0                   |
|                                  | 5-Year EPS CAGR         | 10     | +10 if > 10%; +5 if 5–10%; else 0                   |
| 💵 **4. Valuation (15 pts)**     | P/E vs Industry         | 10     | +10 if < industry avg; +5 if slightly above; else 0 |
|                                  | PEG Ratio               | 5      | +5 if PEG < 1.5; else 0                             |
| 📊 **5. Momentum (15 pts)**      | 200-day vs 50-day MA    | 10     | +10 if price > 200MA & 50MA; else 0                 |
|                                  | Relative Strength (RSI) | 5      | +5 if RSI 40–60 (stable zone); else 0               |
| 🛡️ **6. Safety (10 pts)**       | Beta (Volatility)       | 10     | +10 if β < 1; +5 if 1–1.2; else 0                   |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/StockMentor.git
cd StockMentor
