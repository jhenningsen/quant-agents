import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from hmmlearn.hmm import GaussianHMM
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END

# --- 1. State Definition (JSON-Safe) ---
class State(TypedDict):
    symbols: list
    stock_data: Dict[str, Any]  # Now stores only regime labels: {"AAPL": {"regime": "TRENDING"}}
    analysis_results: list
    final_summary: str

# --- 2. HMM Regime Classifier Node ---
def classify_regime_hmm(state: State):
    regime_results = {}

    # Logic to get symbols from State or CSV
    symbols = state.get("symbols", [])
    if not symbols:
        try:
            df_csv = pd.read_csv("OptionVolume.csv")
            symbols = df_csv["Symbol"].str.strip().tolist()
        except:
            symbols = ["AAPL", "NVDA", "MSFT"]

    for ticker in symbols:
        try:
            # Fetch 3Y data for HMM training depth
            df = yf.download(ticker, period="3y", progress=False)
            if len(df) < 50: continue

            # HMM Feature Engineering
            df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['range'] = (df['High'] - df['Low']) / df['Close']
            features = df[['returns', 'range']].dropna()

            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=42)
            model.fit(features.values)

            states = model.predict(features.values)
            state_vars = [np.var(features.values[states == i, 0]) for i in range(2)]
            trending_state = np.argmin(state_vars)

            regime = "TRENDING" if states[-1] == trending_state else "SIDEWAYS"

            # CRITICAL FIX: Only save the regime label, NOT the DataFrame
            regime_results[ticker] = {"regime": regime}
            print(f"DEBUG: {ticker} classified as {regime}")
        except Exception as e:
            print(f"Error classifying {ticker}: {e}")

    return {"stock_data": regime_results}

# --- 3. Strategy Nodes (Fetch-on-Demand Pattern) ---

def trending_node(state: State):
    results = []
    # Use standard multipliers for a "LinkedIn-Ready" strategy
    L = 10
    M = 2

    for ticker, info in state['stock_data'].items():
        if info['regime'] == "TRENDING":
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty or len(df) < L: continue

            # 1. Calculate Indicators
            st = ta.supertrend(df['High'], df['Low'], df['Close'], length=L, multiplier=M)
            stc = ta.stc(df['Close'])

            if st is None or stc is None: continue

            try:
                curr_price = df['Close'].iloc[-1]
                # DYNAMIC COLUMN SELECTION: Fixes the 'NoneType' and 'Skip' errors
                st_column = f'SUPERT_{L}_{float(M)}'

                if st_column not in st.columns:
                    print(f"DEBUG: Column {st_column} not found in {st.columns.tolist()}")
                    continue

                st_floor = st[st_column].iloc[-1]
                stc_val = stc.iloc[-1]

                # SIGNAL LOGIC
                # Let's use a standard 'Bullish' STC (> 25)
                if curr_price > st_floor and stc_val > 25:
                    results.append({
                        "symbol": ticker,
                        "regime": "TRENDING",
                        "signal": "BUY",
                        "reason": f"Breakout: Price(${curr_price:.2f}) > Floor(${st_floor:.2f}) & STC({stc_val:.1f}) Bullish"
                    })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    return {"analysis_results": results}

def sideways_node(state: State):
    """Mean Reversion Logic with local data fetch"""
    results = []
    for ticker, info in state['stock_data'].items():
        if info['regime'] == "SIDEWAYS":
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty or len(df) < 20: continue

            rsi = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)

            if rsi is None or bb is None: continue

            try:
                curr_rsi = rsi.iloc[-1]
                lower_bb = bb['BBL_20_2.0'].iloc[-1]
                curr_price = df['Close'].iloc[-1]

                if curr_rsi < 35 and curr_price < lower_bb:
                    results.append({
                        "symbol": ticker,
                        "regime": "SIDEWAYS",
                        "signal": "BUY",
                        "reason": "Mean Reversion (Oversold Range)"
                    })
            except Exception as e:
                print(f"Error processing sideways {ticker}: {e}")
    return {"analysis_results": results}

# --- 4. Summarization Node ---
def summarize_hmm_findings(state: State):
    all_res = state.get("analysis_results", [])
    if not all_res:
        return {"final_summary": "HMM Analysis Complete: No actionable setups found."}

    lines = ["### HMM AI-REGIME ANALYSIS SUMMARY ###"]
    for r in all_res:
        lines.append(f"- **{r['symbol']}** ({r['regime']}): {r['signal']} | {r['reason']}")

    return {"final_summary": "\n".join(lines)}

# --- 5. Graph Orchestration ---
workflow = StateGraph(State)
workflow.add_node("classify", classify_regime_hmm)
workflow.add_node("trending_strategy", trending_node)
workflow.add_node("sideways_strategy", sideways_node)
workflow.add_node("summarize", summarize_hmm_findings)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "trending_strategy")
workflow.add_edge("classify", "sideways_strategy")
workflow.add_edge("trending_strategy", "summarize")
workflow.add_edge("sideways_strategy", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()
