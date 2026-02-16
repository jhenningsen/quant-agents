import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# --- 1. Configuration ---
RSI_THRESHOLD = 25
RSI_LENGTHS = [12, 14, 16, 18, 20, 22, 24, 26]
CSV_FILE = "OptionVolume.csv"

class AgentState(TypedDict):
    # We make these optional so the graph can start with {}
    signals: Optional[List[dict]]
    status: Optional[str]

# --- 2. Precision RSI Logic ---
def calculate_rsi_wilder(series, period=14):
    """Matches Yahoo Finance Precision (Wilder's Smoothing)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Use EWM with alpha = 1/period (Wilder's)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- 3. The Scanner Node ---
def rsi_scanner_node(state: AgentState):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Full Scanner...")

    try:
        df_csv = pd.read_csv(CSV_FILE)
        symbol_col = [c for c in df_csv.columns if 'symbol' in c.lower()][0]
        symbols = df_csv[symbol_col].str.strip().unique().tolist()
    except Exception as e:
        return {"signals": [], "status": f"Error: {e}"}

    found_signals = []

    for s in symbols:
        try:
            # PULLING 200 DAYS: Required for RSI math to stabilize and match Yahoo/TradingView
            df = yf.download(s, period="200d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = df['Close'].iloc[-1]

            for length in RSI_LENGTHS:
                rsi_series = calculate_rsi_wilder(df['Close'], period=length)
                rsi_today = rsi_series.iloc[-1]
                rsi_yesterday = rsi_series.iloc[-2]

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

    return {
        "signals": found_signals,
        "status": f"Found {len(found_signals)} signals"
    }

# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("scanner", rsi_scanner_node)
workflow.set_entry_point("scanner")
workflow.add_edge("scanner", END)

# Export as 'graph'
graph = workflow.compile()
