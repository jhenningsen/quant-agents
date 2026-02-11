# Quant Agents: Momentum Continuity Suite
A collection of high-performance quantitative trading agents built with **LangGraph** and **Vectorized Pandas**. These agents identify "Persistence Setups" where short-term momentum aligns with long-term statistical edges.

## 🚀 Featured Agent: 14/10 Momentum Continuity
This agent identifies specific setups where the price is "sandwiched" between a trending floor (SMA 12-14) and a mean-reversion ceiling (Bollinger Band Midpoint).

### Key Features
* **Multi-Period Backtesting:** Every signal is automatically verified against 1-Year, 2-Year, and 3-Year historical windows to detect **Alpha Decay**.
* **Multi-Horizon Returns:** Calculates mean forward returns for **3-Day**, **5-Day**, and **10-Day** periods to assess trade momentum.
* **Dynamic Optimization:** Iterates through a range of Moving Averages to find the specific "Floor" with the highest 3-year win rate.
* **Stateful Orchestration:** Built on LangGraph to ensure resilient data flow and structured reporting.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (Stateful Agents)
* **Data:** YFinance, Pandas
* **Technical Analysis:** Pandas_TA (Vectorized Indicator Logic)

## 📋 Requirements
To run the agents in this repository, you need the following Python libraries:

```bash
pip install pandas pandas_ta yfinance langgraph

