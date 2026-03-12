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

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    tools=[{"google_search_grounding": {}}]
)

# --- 1. Configuration (Power Zone Logic) ---
RSI_THRESHOLD = 80  # Scanning for Overbought/Momentum Ignition
RSI_LENGTHS = [7, 10] # Focused on medium-term momentum
CSV_FILE = "OptionVolume.csv"

# --- 2. State Definition ---
class AgentState(TypedDict):
    signals: Optional[List[dict]]
    final_report: Optional[str]
    status: Optional[str]

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
            df = yf.download(s, period="200d", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 30: continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            current_close = float(df['Close'].iloc[-1])
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]

            rsi_matches = []
            for length in RSI_LENGTHS:
                rsi_series = calculate_rsi_wilder(df['Close'], period=length)
                rsi_today = float(rsi_series.iloc[-1])
                rsi_yesterday = float(rsi_series.iloc[-2])

                # TRIGGER: Just entered the Power Zone (> 70) from below
                if rsi_today > RSI_THRESHOLD:
                    rsi_matches.append({"len": length, "val": round(rsi_today, 2)})

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
            f"INSTRUCTION: Use GOOGLE SEARCH to find the next earnings date for {ticker}. "
            f"\n\nDATA: {ticker} just triggered a Momentum Power Zone signal (RSI > 80) at {rsi_summary}. "
            f"Analyze if this is a 'FOMO top' or a legitimate 'Institutional Breakout.' "
            f"\n\nREQUIRED OUTPUT FORMAT:"
            f"\nNEXT EARNINGS: [Confirmed or Estimated Date]"
            f"\nAnalysis: 3-sentence summary regarding recent news catalysts and if "
            f"this level has historically led to further expansion for {ticker}."
        )

        response = llm.invoke(prompt)
        res_content = response.content[0].get('text', '') if isinstance(response.content, list) else response.content
        item['ai_insight'] = res_content.strip()
        enriched.append(item)

    return {"signals": enriched, "status": "Momentum Research Complete"}

# --- 6. Summarize Node ---
def summarize_node(state: AgentState):
    signals = state.get("signals", [])
    if not signals:
        return {"final_report": "No momentum breakouts found today."}

    report = "## 🔥 RSI MOMENTUM POWER ZONE REPORT\n"
    report += "> *Signals triggered by RSI crossing ABOVE 80 (Initial Momentum Blast)*\n\n"

    for s in signals:
        insight = s.get('ai_insight', 'Pending...')
        lines = insight.split('\n')
        earnings_line = lines[0] if "NEXT EARNINGS" in lines[0] else "📅 Next Earnings: N/A"
        cleaned_insight = "\n".join(lines[1:]) if "NEXT EARNINGS" in lines[0] else insight

        report += f"### 🚀 {s['symbol']} | Price: ${s['price']} | Trend: {s['trend']}\n"
        report += f"**{earnings_line}**\n"
        rsi_pairs = ", ".join([f"**L{m['len']}**: {m['val']}" for m in s.get('rsi_matches', [])])
        report += f"⚡ **Momentum Strength:** {rsi_pairs}\n\n"
        report += f"**AI Momentum Analysis:**\n> {cleaned_insight.strip()}\n\n"
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
