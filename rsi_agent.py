import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import yfinance as yf
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file
load_dotenv()

# LangChain will now automatically find os.environ["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1
)

# --- 1. Configuration ---
RSI_THRESHOLD = 27
RSI_LENGTHS = [10, 12, 14, 16, 18, 22, 26]
CSV_FILE = "OptionVolume.csv"


# --- 2. State Definition (JSON-Safe) ---
class AgentState(TypedDict):
    # 'signals' will hold the list of stocks that passed the RSI scan
    signals: Optional[List[dict]]
    # 'final_report' will hold the AI-formatted summary
    final_report: Optional[str]
    status: Optional[str]

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Multi-Length Scanner...")

    try:
        df_csv = pd.read_csv(CSV_FILE)
        symbol_col = [c for c in df_csv.columns if 'symbol' in c.lower()][0]
        symbols = df_csv[symbol_col].str.strip().unique().tolist()
    except Exception as e:
        return {"signals": [], "status": f"Error: {e}"}

    found_signals = []

    for s in symbols:
        try:
            # 1. Download Price Data
            df = yf.download(s, period="200d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = float(df['Close'].iloc[-1])
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]

            # 2. Extract Earnings Date (Source of Truth)
            earnings_date = "N/A"
            try:
                ticker_info = yf.Ticker(s)
                # Check info dict first (it's often more reliable than .calendar)
                earnings_ts = ticker_info.info.get('nextEarningsDate')
                if earnings_ts:
                    earnings_date = datetime.fromtimestamp(earnings_ts).strftime('%Y-%m-%d')
                else:
                    # Fallback to calendar
                    cal = ticker_info.calendar
                    if cal is not None and not cal.empty:
                        earnings_date = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
            except:
                earnings_date = "Check Nasdaq"

            # 3. Check RSI across all lengths
            rsi_matches = []
            for length in RSI_LENGTHS:
                rsi_series = calculate_rsi_wilder(df['Close'], period=length)
                rsi_today = float(rsi_series.iloc[-1])

                if rsi_today < RSI_THRESHOLD:
                    rsi_matches.append({"len": length, "val": round(rsi_today, 2)})

            # 4. Compile Signal
            if rsi_matches:
                found_signals.append({
                    "symbol": s,
                    "price": round(current_close, 2),
                    "earnings_date": earnings_date, # Passed to the Research Node
                    "trend": "Bullish" if current_close > sma_200 else "Bearish",
                    "rsi_matches": rsi_matches,
                    "rsi_val": rsi_matches[0]['val']
                })
        except:
            continue

    return {
        "signals": found_signals,
        "status": f"Found {len(found_signals)} tickers with signals"
    }

# --- 5. Separate AI Research Node ---
def research_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"status": "No signals found to research."}

    enriched = []
    for item in signals:
        ticker = item['symbol']

        # Convert the list of matches into a readable string for the AI
        rsi_summary = ", ".join([f"L{m['len']}: {m['val']}" for m in item.get('rsi_matches', [])])
        actual_date = item.get('earnings_date', 'N/A')

        prompt = (
            f"Act as a quantitative analyst. Analyze {ticker}. "
            f"FACT: The next earnings date is {actual_date}. "
            f"Current RSI triggers: {rsi_summary}. "
            f"Summarize market sentiment and the historical risk of buying this RSI dip in 3 sentences. "
            f"Do not guess the earnings date; use the one provided."
        )

        response = llm.invoke(prompt)
        if isinstance(response.content, list):
            # Dig into the list -> first dictionary -> 'text' key
            raw_text = response.content[0].get('text', '')
        else:
            # If it's already a string (fallback)
            raw_text = response.content

        item['ai_insight'] = raw_text.strip()
        enriched.append(item)

    return {"signals": enriched, "status": "Research Complete"}

# --- 6. Summarize Node (The Presentation Layer) ---
def summarize_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"final_report": "No actionable RSI signals detected today."}

    report = "## 📊 RSI QUANT RESEARCH REPORT\n\n"

    for s in signals:
        report += f"### 🔍 {s['symbol']} | Price: ${s['price']}\n"

        # Display Yahoo Finance Earnings Date clearly
        report += f"📅 **Next Earnings:** {s.get('earnings_date', 'N/A')}\n"

        # Display the RSI pairs
        rsi_pairs = ", ".join([f"**L{m['len']}**: {m['val']}" for m in s.get('rsi_matches', [])])
        report += f"📉 **Oversold Triggers:** {rsi_pairs}\n\n"

        # Display the AI Insight
        insight = s.get('ai_insight', 'Research pending...')
        report += f"**AI Analysis:**\n> {insight}\n\n"
        report += "---\n"

    print(report)
    return {"final_report": report}

# --- 7. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("scanner", rsi_scanner_node)
workflow.add_node("researcher", research_node)
workflow.add_node("summarizer", summarize_node)

workflow.set_entry_point("scanner")
workflow.add_edge("scanner", "researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", END)

graph = workflow.compile()
