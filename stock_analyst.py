import pandas as pd

def generate_summary(df, ticker):
    """
    Generates a plain English analysis of the stock based on technical indicators.
    Returns a markdown string.
    """
    if df.empty:
        return "No data available for analysis."
        
    # Get latest row
    latest = df.iloc[-1]
    
    summary = []
    summary.append(f"### 🧐 Tech Analysis for **{ticker}**")
    
    # 1. Price Trend (SMA 50 vs 200)
    if 'SMA_50' in latest and 'SMA_200' in latest:
        price = latest['Close']
        sma50 = latest['SMA_50']
        sma200 = latest['SMA_200']
        
        if price > sma200:
            trend = "📈 **Bullish (Up)**"
            desc = "The price is above the 200-day average, indicating a long-term upward trend."
        else:
            trend = "📉 **Bearish (Down)**"
            desc = "The price is below the 200-day average, indicating a long-term downward trend."
            
        summary.append(f"- **Long-Term Trend**: {trend}. {desc}")
        
        if sma50 > sma200:
            summary.append(f"- **Golden Cross**: The 50-day average is above the 200-day average. This is generally a very positive sign.")
        elif sma50 < sma200:
            summary.append(f"- **Death Cross**: The 50-day average is below the 200-day average. This is generally a negative sign.")

    # 2. Momentum (RSI)
    if 'RSI' in latest:
        rsi = latest['RSI']
        if rsi > 70:
            rsi_status = "🔥 **Overbought**"
            rsi_desc = "The stock might be too expensive right now. It could dip soon."
        elif rsi < 30:
            rsi_status = "🧊 **Oversold**"
            rsi_desc = "The stock has been sold off heavily. It might bounce back soon."
        else:
            rsi_status = "⚖️ **Neutral**"
            rsi_desc = "The stock is neither overbought nor oversold."
            
        summary.append(f"- **Momentum (RSI)**: {rsi_status} ({rsi:.1f}). {rsi_desc}")

    # 3. MACD
    if 'MACD' in latest and 'MACD_Signal' in latest:
        macd = latest['MACD']
        signal = latest['MACD_Signal']
        
        if macd > signal:
            macd_status = "🟢 **Positive**"
            macd_desc = "The MACD line is above the signal line. This suggests buying pressure."
        else:
            macd_status = "🔴 **Negative**"
            macd_desc = "The MACD line is below the signal line. This suggests selling pressure."
            
        summary.append(f"- **MACD Sentiment**: {macd_status}. {macd_desc}")

    # 4. Bollinger Bands
    if 'BB_Upper' in latest and 'BB_Lower' in latest:
        price = latest['Close']
        upper = latest['BB_Upper']
        lower = latest['BB_Lower']
        
        if price >= upper:
            bb_status = "⚠️ **Touching Top Price**"
            bb_desc = "Price is hitting the upper Bollinger Band. It might be overextended."
            summary.append(f"- **Volatility Bands**: {bb_status}. {bb_desc}")
        elif price <= lower:
            bb_status = "⚠️ **Touching Bottom Value**"
            bb_desc = "Price is hitting the lower Bollinger Band. It might be undervalued."
            summary.append(f"- **Volatility Bands**: {bb_status}. {bb_desc}")

    # 5. OBV (Volume)
    if 'OBV' in latest:
        # Check slope of last 5 days
        recent_obv = df['OBV'].tail(5)
        if len(recent_obv) > 1:
            if recent_obv.iloc[-1] > recent_obv.iloc[0]:
                obv_status = "Volume is **Increasing**"
            else:
                obv_status = "Volume is **Decreasing**"
            summary.append(f"- **Volume Flow**: {obv_status} over the last 5 days.")

    return "\n\n".join(summary)
