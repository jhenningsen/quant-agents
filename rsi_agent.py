import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END

# --- 1. Configuration ---
RSI_THRESHOLD = 25
RSI_LENGTHS = [12, 14, 16, 18, 20, 22, 24, 26]
CSV_FILE = "OptionVolume.csv"

# --- 2. State Definition ---
class AgentState(TypedDict):
    """The state of our RSI scanner agent."""
    signals: List[dict]
    status: str

# --- 3. Helper Logic ---
def calculate_rsi_yahoo(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- 4. The Scanner Node ---
def rsi_scanner_node(state: AgentState):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] RSI Agent starting scan...")

    try:
        df_csv = pd.read_csv(CSV_FILE)
        symbol_col = [c for c in df_csv.columns if 'symbol' in c.lower()][0]
        symbols = df_csv[symbol_col].str.strip().unique().tolist()
    except Exception as e:
        return {"signals": [], "status": f"Error loading CSV: {e}"}

    found_signals = []

    for s in symbols:
        try:
            # Fetch data
            df = yf.download(s, period="60d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 30: continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = df['Close'].iloc[-1]

            for length in RSI_LENGTHS:
                rsi_series = calculate_rsi_yahoo(df['Close'], period=length)
                rsi_today = rsi_series.iloc[-1]
                rsi_yesterday = rsi_series.iloc[-2]

                # Check for Threshold 25 Cross or Hold
                if rsi_today < RSI_THRESHOLD:
                    signal_type = "CROSS_DOWN (NEW)" if rsi_yesterday >= RSI_THRESHOLD else "OVERSOLD"

                    found_signals.append({
                        "Symbol": s,
                        "Price": round(float(current_close), 2),
                        "RSI_Len": length,
                        "RSI_Val": round(float(rsi_today), 2),
                        "Signal": signal_type
                    })
        except:
            continue

    # Save to CSV for external review
    if found_signals:
        filename = f"rsi_signals_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame(found_signals).to_csv(filename, index=False)

    return {
        "signals": found_signals,
        "status": "Success" if found_signals else "No signals found"
    }

# --- 5. Graph Construction ---
# Define the graph
workflow = StateGraph(AgentState)

# Add the scanner node
workflow.add_node("scanner", rsi_scanner_node)

# Set the entry point and exit point
workflow.set_entry_point("scanner")
workflow.add_edge("scanner", END)

# THE CRITICAL LINE: Export the 'graph' variable for LangGraph API
graph = workflow.compile()

# --- 6. Standalone Testing ---
# This allows you to still run 'python rsi_agent.py' manually
if __name__ == "__main__":
    print("Running RSI Agent manually...")
    result = graph.invoke({"signals": [], "status": "starting"})
    print("\nSignals Found:", len(result['signals']))
    if result['signals']:
        print(pd.DataFrame(result['signals']).to_string(index=False))
