import pandas as pd
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file
load_dotenv()

# LangChain will now automatically find os.environ["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- 1. Configuration ---
RSI_THRESHOLD = 25
RSI_LENGTHS = [10, 12, 14, 16, 18, 22, 26]
CSV_FILE = "OptionVolume.csv"


# --- 2. State Definition (JSON-Safe) ---
class AgentState(TypedDict):
    # 'signals' will hold the list of stocks that passed the RSI scan
    signals: Optional[List[dict]]
    # 'final_report' will hold the AI-formatted summary
    final_report: Optional[str]
    status: Optional[str]

# Initialize AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- 3. Precision RSI Logic ---
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

# --- 4. The Scanner Node ---
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

            # Optional: Keep SMA 200 just as a data point, not a filter
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            trend = "Bullish" if current_close > sma_200 else "Bearish"

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

# --- 5. Separate AI Research Node ---
def research_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"status": "No signals found to research."}

    enriched = []
    for item in signals:
        ticker = item['symbol']
        # Call Gemini for the 'AI Element'
        prompt = f"Analyze {ticker}. It is currently at an RSI of {item['rsi']}. Briefly mention one upcoming catalyst (like earnings) and the historical risk of buying this dip."
        response = llm.invoke(prompt)

        item['ai_insight'] = response.content
        enriched.append(item)

    return {"signals": enriched, "status": "Research Complete"}

# --- 6. Summarize Node (The Presentation Layer) ---
def summarize_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"final_report": "No actionable RSI signals detected today."}

    # Creating a structured text summary
    report = "### 📈 RSI SCANNER SUMMARY\n\n"
    for s in signals:
        report += f"**{s['symbol']}** (RSI: {s['rsi']:.1f})\n"
        report += f"- Trend: {s['trend']} | Price: ${s['price']}\n"
        report += f"- AI Insight: {s['ai_insight']}\n\n"

    print(report) # Also print to console
    return {"final_report": report, "status": "Finished"}

# --- 7. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("scanner", rsi_scanner_node)
workflow.add_node("researcher", research_node)
workflow.add_node("summarizer", summarize_node)

workflow.set_entry_point("scanner")
workflow.add_edge("scanner", "researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()
