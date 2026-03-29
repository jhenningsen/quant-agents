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

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    tools=[{"google_search_grounding": {}}]
)

# --- 1. Configuration (Power Zone Logic) ---
# Format: (Length, Threshold)
RSI_CONFIG = [
    (10, 80),
    (14, 75)
]
CSV_FILE = "OptionVolume.csv"

# --- 2. State Definition ---
class AgentState(TypedDict):
    signals: Optional[List[dict]]
    final_report: Optional[str]
    status: Optional[str]
    run_id: Optional[str]

# --- 3. Precision RSI Logic ---
def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- 4. The Power Zone Scanner Node ---
def rsi_scanner_node(state: AgentState):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for Power Zone Breakouts...")

    try:
        df_csv = pd.read_csv(CSV_FILE)
        symbol_col = [c for c in df_csv.columns if 'symbol' in c.lower() or 'ticker' in c.lower()][0]
        symbols = df_csv[symbol_col].str.strip().unique().tolist()
    except Exception as e:
        return {"signals": [], "status": f"Error loading CSV: {e}"}

    found_signals = []

    for idx, s in enumerate(symbols):
        try:
            # Need at least 2 days to check for crossover
            df = yf.download(s, period="300d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 30: continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = float(df['Close'].iloc[-1])
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]

            # 2. Check RSI across configured Momentum Pairs
            rsi_matches = []
            for length, threshold in RSI_CONFIG:
                rsi_series = calculate_rsi_wilder(df['Close'], period=length)
                rsi_today = float(rsi_series.iloc[-1])
                rsi_yesterday = float(rsi_series.iloc[-2])

                # TRIGGER: Momentum Ignition (Crossing ABOVE threshold)
                if rsi_today > threshold and rsi_yesterday <= threshold:
                    rsi_matches.append({
                        "len": length,
                        "val": round(rsi_today, 2),
                        "threshold": threshold
                    })

            if rsi_matches:
                found_signals.append({
                    "symbol": s,
                    "price": round(current_close, 2),
                    "trend": "Strong Bullish" if current_close > sma_200 else "Recovery",
                    "position": idx + 1,
                    "rsi_matches": rsi_matches
                })
        except:
            continue

    return {"signals": found_signals, "status": f"Found {len(found_signals)} momentum triggers"}

# --- 5. Momentum Research Node ---
def research_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals: return {"status": "No signals."}

    current_date_str = datetime.now().strftime('%B %d, %Y')
    enriched = []

    for item in signals:
        ticker = item['symbol']
        rsi_summary = ", ".join([f"L{m['len']}: {m['val']}" for m in item.get('rsi_matches', [])])

        prompt = (
            f"SYSTEM: Current date is {current_date_str}. "
            f"INSTRUCTION: You are a financial analyst. Use GOOGLE SEARCH to find: "
            f"LAST earnings date for {ticker} (this should be before the {current_date_str}). "
            f"IGNORE all dates after the {current_date_str}. "
            f"NEXT UPCOMING earnings date for {ticker} (this should be after the {current_date_str}). "
            f"IGNORE all dates before the {current_date_str}. "
            f"\n\nDATA: {ticker} just triggered a Momentum Power Zone signal with triggers: {rsi_summary}. "
            f"Analyze the likelihood of further price appreciation "
            f"\n\nREQUIRED OUTPUT FORMAT:"
            f"\nNEXT EARNINGS: [Date]"
            f"\nLAST EARNINGS: [Date]"
            f"\nAnalysis: 4-sentence summary regarding recent news catalysts and market sentiment. Also include "
            f"analysis on whether this RSI level has historically led to further expansion for {ticker}."
        )

        response = llm.invoke(prompt)
        res_content = response.content[0].get('text', '') if isinstance(response.content, list) else response.content
        item['ai_insight'] = res_content.strip()
        enriched.append(item)

    return {"signals": enriched, "status": "Momentum Research Complete"}

def summarize_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"final_report": "No momentum breakouts found today."}

    report = "## 🔥 RSI MOMENTUM POWER ZONE REPORT\n\n"

    for s in signals:
        insight = s.get('ai_insight', 'Pending...')
        lines = insight.split('\n')

        # 1. Extract specific lines
        next_e = next((l for l in lines if "NEXT EARNINGS" in l), "📅 Next Earnings: N/A")
        last_e = next((l for l in lines if "LAST EARNINGS" in l), "⏪ Last Earnings: N/A")

        # 2. Extract the Analysis text specifically
        # We look for the line starting with "Analysis:" and remove that prefix
        analysis_line = next((l for l in lines if l.startswith("Analysis:")), "")
        if not analysis_line:
            # Fallback: if AI didn't label it, just take the last few lines that aren't dates
            analysis_text = "\n".join([l for l in lines if "EARNINGS" not in l]).strip()
        else:
            analysis_text = analysis_line.replace("Analysis:", "").strip()

        report += f"### 🚀 {s['symbol']} | Price: ${s['price']} | Trend: {s['trend']}\n"
        report += f"**{next_e}**\n"
        report += f"**{last_e}**\n"

        rsi_pairs = ", ".join([f"**L{m['len']}**: {m['val']} (Blast > {m['threshold']})" for m in s.get('rsi_matches', [])])
        report += f"⚡ **Momentum Strength:** {rsi_pairs}\n\n"
        report += f"**AI Momentum Analysis:**\n> {analysis_text}\n\n"
        report += "---\n"

    return {"final_report": report}

# --- 7. Build Graph ---
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
if __name__ == "__main__":
    # Every time you run this, a unique thread ID is created based on the time
    unique_thread_id = f"momentum_run_{int(time.time())}"
    config = {"configurable": {"thread_id": unique_thread_id}}

    print(f"--- Running Momentum Agent with Thread ID: {unique_thread_id} ---")

    # Run the graph and print results
    # We pass an empty signals list and the unique run_id to initialize
    result = graph.invoke({"signals": [], "run_id": unique_thread_id}, config)

    if "final_report" in result:
        print(result["final_report"])
