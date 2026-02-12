import pandas as pd
import yfinance as yf
import pandas_ta as ta
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    symbols: list
    sma_min: int
    sma_max: int
    analysis_results: list
    final_summary: str

def analyze_stocks(state: State):
    scorecard = []

    # 1. Parameters - define the range
    s_min = state.get("sma_min", 12)
    s_max = state.get("sma_max", 14)
    sma_range = range(s_min, s_max + 1)
    bb_length = 10

    # 2. Symbol Selection (Logic remains same)
    if state.get("symbols") and len(state["symbols"]) > 0:
        symbol_list = [s.strip() for s in state["symbols"]]
    else:
        try:
            df_csv = pd.read_csv("OptionVolume.csv")
            symbol_list = df_csv["Symbol"].str.strip().tolist()
        except Exception as e:
            return {"analysis_results": [{"error": f"CSV Error: {e}"}]}

    for sym in symbol_list:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="3y") # Ensure 3y for backtest depth

            if len(df) < 20: # Basic safety check
                continue

            # BB 10 Midpoint acts as the "Ceiling"
            bbands = ta.bbands(df['Close'], length=bb_length, std=2)
            bbm_col = [col for col in bbands.columns if col.startswith('BBM')][0]

            curr = df['Close'].iloc[-1]
            midpoint_ceiling = bbands[bbm_col].iloc[-1]

            best_history = None
            triggered_sma = None
            rec = "WATCH"

            # --- RANGE LOGIC: Check 12, 13, and 14 ---
            # Calculate all SMAs at once (Pandas is optimized for this)
            for s_val in sma_range:
                df[f'SMA_{s_val}'] = ta.sma(df['Close'], length=s_val)

            # Identify the "Best" SMA for the current price action
            # We want to see which of our 12, 13, or 14 SMA is acting as the best 'Floor'
            for s_val in sma_range:
                sma_col = f'SMA_{s_val}'
                trend_floor = df[sma_col].iloc[-1]
                prev_floor = df[sma_col].iloc[-2]  # Yesterday's SMA
                prev_close = df['Close'].iloc[-2]  # Yesterday's Close

                # 1. NEW TRIGGER LOGIC: Must cross FROM below TO above the floor
                was_below_yesterday = prev_close < prev_floor
                is_above_today = curr > trend_floor
                below_ceiling = curr < midpoint_ceiling

                if was_below_yesterday and is_above_today and below_ceiling:
                    rec = "BUY"
                    hist_df = df.copy()

                    # 2. NEW BACKTEST LOGIC: Vectorized crossover check
                    # Compare each day's close to its SMA, and the previous day's close to its SMA
                    hist_df['prev_close'] = hist_df['Close'].shift(1)
                    hist_df['prev_sma'] = hist_df[sma_col].shift(1)

                    hist_df['sig'] = (hist_df['prev_close'] < hist_df['prev_sma']) & \
                                     (hist_df['Close'] > hist_df[sma_col]) & \
                                     (hist_df['Close'] < bbands[bbm_col])

                    # Calculate Multiple Forward Return Windows
                    hist_df['ret_3d'] = hist_df['Close'].shift(-3) / hist_df['Close'] - 1
                    hist_df['ret_5d'] = hist_df['Close'].shift(-5) / hist_df['Close'] - 1
                    hist_df['ret_10d'] = hist_df['Close'].shift(-10) / hist_df['Close'] - 1

                    stats = {}
                    periods = {"1Y": 252, "2Y": 504, "3Y": len(df)}

                    for label, days in periods.items():
                        # Dropna ensures we only count signals where the future dates actually exist
                        slice_df = hist_df.tail(days).dropna(subset=['ret_10d'])
                        signals = slice_df[slice_df['sig'] == True]

                        if not signals.empty:
                            w_rate = len(signals[signals['ret_10d'] > 0]) / len(signals)
                            stats[label] = {
                                "win_rate": f"{w_rate:.1%}",
                                "count": len(signals),
                                "avg_3d": f"{signals['ret_3d'].mean():.2%}",
                                "avg_5d": f"{signals['ret_5d'].mean():.2%}",
                                "avg_10d": f"{signals['ret_10d'].mean():.2%}",
                                "wr_num": w_rate
                            }
                        else:
                            stats[label] = {"win_rate": "N/A", "count": 0, "wr_num": 0}

                    if best_history is None or stats["3Y"]["wr_num"] > best_history["3Y"]["wr_num"]:
                        best_history = stats
                        triggered_sma = s_val

            # --- 3. UPDATED SCORECARD ---
            scorecard.append({
                "symbol": sym,
                "price": round(curr, 2),
                "bb_ceiling": round(midpoint_ceiling, 2),
                "triggered_sma": triggered_sma,
                "multi_period": best_history if best_history else {"note": "N/A"},
                "recommendation": rec
            })

        except Exception as e:
            scorecard.append({"symbol": sym, "status": "Error", "note": str(e)})

    return {"analysis_results": scorecard}

def summarize_findings(state: State):
    buy_results = [res for res in state['analysis_results'] if res.get('recommendation') == "BUY"]

    if not buy_results:
        # Returning a dictionary, not just a string
        return {"final_summary": "No stocks currently meet criteria."}

    lines = ["### Multi-Period Performance Analysis ###"]

    for res in buy_results:
        sym = res['symbol']
        mp = res.get('multi_period')

        if not mp:
            lines.append(f"- **{sym}** @ ${res['price']} (SMA {res['triggered_sma']})\n  └ No historical signals found.")
            continue

        lines.append(f"- **{sym}** @ ${res['price']} (SMA {res['triggered_sma']})")

        for period in ["1Y", "2Y", "3Y"]:
            data = mp.get(period)
            if data and data.get("count", 0) > 0:
                lines.append(f"  └ {period} ({data['count']} sigs): WR {data['win_rate']} | Avg Ret: [3D: {data['avg_3d']}, 5D: {data['avg_5d']}, 10D: {data['avg_10d']}]")

    # The fix: wrap the joined string in a dictionary matching the State key
    return {"final_summary": "\n".join(lines)}

workflow = StateGraph(State)
workflow.add_node("analyze_stocks", analyze_stocks)
workflow.add_node("summarize", summarize_findings)
workflow.add_edge(START, "analyze_stocks")
workflow.add_edge("analyze_stocks", "summarize")
workflow.add_edge("summarize", END)
graph = workflow.compile()
