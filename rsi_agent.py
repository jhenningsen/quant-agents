import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import yfinance as yf
import numpy as np
import time
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
    tools=[{"google_search_grounding": {}}] # This allows the AI to search the web
)

# --- 1. Configuration ---
# Format: (Length, Threshold)
RSI_CONFIG = [
    (16, 25),
    (24, 30),
    (14, 25),
    (22, 30),
    (18, 25),
    (22, 25),
    (26, 30)
]
CSV_FILE = "OptionVolume.csv"

# --- 2. State Definition (JSON-Safe) ---
class AgentState(TypedDict):
    # 'signals' will hold the list of stocks that passed the RSI scan
    signals: Optional[List[dict]]
    # 'final_report' will hold the AI-formatted summary
    final_report: Optional[str]
    status: Optional[str]
    # 'run_id' will act as a cache-buster
    run_id: Optional[str]

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Lean Multi-Length Scanner...")

    try:
        df_csv = pd.read_csv(CSV_FILE)
        symbol_col = [c for c in df_csv.columns if 'symbol' in c.lower()][0]
        symbols = df_csv[symbol_col].str.strip().unique().tolist()
    except Exception as e:
        return {"signals": [], "status": f"Error: {e}"}

    found_signals = []

    for idx, s in enumerate(symbols):
        try:
            # 1. Download Price Data
            df = yf.download(s, period="300d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = float(df['Close'].iloc[-1])
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]

            # 2. Check RSI across configured pairs
            rsi_matches = []
            for length, threshold in RSI_CONFIG:
                rsi_series = calculate_rsi_wilder(df['Close'], period=length)
                rsi_today = float(rsi_series.iloc[-1])

                # Evaluate current RSI against its specific paired threshold
                if rsi_today < threshold:
                    rsi_matches.append({
                        "len": length,
                        "val": round(rsi_today, 2),
                        "threshold": threshold  # Optional: carry threshold for AI context
                    })

            # 3. Compile Signal
            if rsi_matches:
                signal_data = {
                    "symbol": s,
                    "price": round(current_close, 2),
                    "trend": "Bullish" if current_close > sma_200 else "Bearish",
                    "position": idx + 1,
                    "rsi_matches": rsi_matches,
                    "rsi_val": rsi_matches[0]['val']
                }
                found_signals.append(signal_data)
        except:
            continue

    return {
        "signals": found_signals,
        "status": f"Found {len(found_signals)} tickers with signals"
    }

# --- 5. Separate AI Research Node ---
def research_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals: return {"status": "No signals."}

    current_date_str = datetime.now().strftime('%B %d, %Y')

    enriched = []
    for item in signals:
        ticker = item['symbol']
        rsi_summary = ", ".join([f"L{m['len']} (Val: {m['val']} < Thresh: {m['threshold']})" for m in item.get('rsi_matches', [])])

        # PROMPT UPDATED FOR SEARCH GROUNDING
        prompt = (
            f"SYSTEM: The current date is {current_date_str}.  "
            f"INSTRUCTION: You are a financial analyst. Use GOOGLE SEARCH to find the "
            f"LAST earnings date for {ticker} (this should be before the {current_date_str}). "
            f"IGNORE all dates after the {current_date_str}. "
            f"NEXT UPCOMING earnings date for {ticker} (this should be after the {current_date_str}). "
            f"IGNORE all dates before the {current_date_str}. "
            f"\n\nDATA: {ticker} is currently oversold with RSI triggers: {rsi_summary}. "
            f"\n\nREQUIRED OUTPUT FORMAT:"
            f"\nLAST EARNINGS: [Confirmed or Estimated Date]"
            f"\nNEXT EARNINGS: [Confirmed or Estimated Date]"
            f"\nAnalysis: A 3-sentence summary of current market sentiment with a focus on why the stock "
            f"is down over the last few days and the historical risk/reward of buying this RSI level for {ticker}."
        )

        response = llm.invoke(prompt)

        # Data extraction safety (handling list vs string)
        res_content = response.content[0].get('text', '') if isinstance(response.content, list) else response.content
        item['ai_insight'] = res_content.strip()
        enriched.append(item)

    return {"signals": enriched, "status": "Research with Search Grounding Complete"}

# --- 6. Summarize Node (The Presentation Layer) ---
def summarize_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"final_report": "No signals found."}

    report = "## 📈 RSI QUANT REPORT (GROUNDED SEARCH)\n\n"

    for s in signals:
        insight = s.get('ai_insight', 'Pending...')
        lines = insight.split('\n')

        # Extract specific lines with safety fallbacks
        next_e = next((l for l in lines if "NEXT EARNINGS" in l), "📅 Next Earnings: N/A")
        last_e = next((l for l in lines if "LAST EARNINGS" in l), "⏪ Last Earnings: N/A")

        # Specifically grab the analysis line and remove the "Analysis:" prefix
        analysis_line = next((l for l in lines if "Analysis:" in l), "")
        cleaned_insight = analysis_line.replace("Analysis:", "").strip()

        # If AI didn't use the "Analysis:" prefix, grab the non-date lines
        if not cleaned_insight:
            cleaned_insight = "\n".join([l for l in lines if "EARNINGS" not in l]).strip()

        report += f"### 🔍 {s['symbol']} | Price: ${s['price']} | Option Volume Rank: #{s['position']}\n"
        report += f"**{next_e}**\n"
        report += f"**{last_e}**\n" # Added display for last earnings

        rsi_pairs = ", ".join([f"**L{m['len']}**: {m['val']}" for m in s.get('rsi_matches', [])])
        report += f"📉 **RSI Triggers:** {rsi_pairs}\n\n"
        report += f"**AI Analysis:**\n> {cleaned_insight.strip()}\n\n"
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

# --- 8. Execution Block (THREAD ID STRATEGY) ---
# This is the primary way to prevent caching
if __name__ == "__main__":
    # Every time you run this, a unique thread ID is created based on the time
    unique_thread_id = f"run_{int(time.time())}"
    config = {"configurable": {"thread_id": unique_thread_id}}

    print(f"--- Running Graph with Thread ID: {unique_thread_id} ---")

    # Run the graph and print results
    result = graph.invoke({"signals": [], "run_id": unique_thread_id}, config)
    if "final_report" in result:
        print(result["final_report"])
