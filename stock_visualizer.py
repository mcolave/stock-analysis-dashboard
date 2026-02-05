import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_chart(df, ticker):
    """
    Generates a Plotly figure for the given dataframe and ticker.
    """
    # Filter for ticker if not already filtered
    if 'Ticker' in df.columns:
        ticker_df = df[df['Ticker'] == ticker].copy()
    else:
        ticker_df = df.copy()
        
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
    
    # Create figure with secondary y-axis for RSI
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Stock Price', 'RSI'),
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=ticker_df['Date'],
                    open=ticker_df['Open'],
                    high=ticker_df['High'],
                    low=ticker_df['Low'],
                    close=ticker_df['Close'],
                    name='OHLC'), row=1, col=1)

    # SMA 50 & 200
    if 'SMA_50' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    if 'SMA_200' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)

    # Bollinger Bands
    if 'BB_Upper' in ticker_df.columns and 'BB_Lower' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper', legendgroup='BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='BB Lower', legendgroup='BB'), row=1, col=1)

    # RSI
    if 'RSI' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
        
    # RSI Reference Lines (70 and 30)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Layout styling
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark"
    )
    
    return fig
