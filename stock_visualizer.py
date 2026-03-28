import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

def create_chart(df, ticker, overlays=None, subplots=None):
    """
    Generates a Plotly figure for the given dataframe and ticker with dynamic overlays and subplots.
    """
    if overlays is None:
        overlays = ['SMA 50', 'SMA 200', 'Bollinger Bands']
    if subplots is None:
        subplots = ['RSI', 'MACD']
        
    # Filter for ticker if not already filtered
    if 'Ticker' in df.columns:
        ticker_df = df[df['Ticker'] == ticker].copy()
    else:
        ticker_df = df.copy()
        
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
    
    # Create dynamic subplots
    num_subplots = len(subplots)
    if num_subplots > 0:
        row_heights = [0.6] + [0.4 / num_subplots] * num_subplots
        fig = make_subplots(rows=1 + num_subplots, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            subplot_titles=[f'{ticker} Stock Price'] + subplots,
                            row_heights=row_heights)
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'{ticker} Stock Price'])

    # Candlestick
    fig.add_trace(go.Candlestick(x=ticker_df['Date'],
                    open=ticker_df['Open'],
                    high=ticker_df['High'],
                    low=ticker_df['Low'],
                    close=ticker_df['Close'],
                    name='OHLC'), row=1, col=1)

    # Overlays
    if 'SMA 50' in overlays and 'SMA_50' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    if 'SMA 200' in overlays and 'SMA_200' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)
    if 'EMA 20' in overlays and 'EMA_20' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
    if 'EMA 50' in overlays and 'EMA_50' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['EMA_50'], line=dict(color='magenta', width=1), name='EMA 50'), row=1, col=1)
        
    if 'Bollinger Bands' in overlays and 'BB_Upper' in ticker_df.columns and 'BB_Lower' in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper', legendgroup='BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='BB Lower', legendgroup='BB'), row=1, col=1)

    # Subplots
    for idx, sp in enumerate(subplots):
        curr_row = idx + 2
        
        if sp == 'RSI' and 'RSI' in ticker_df.columns:
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=curr_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
            
        elif sp == 'MACD' and 'MACD' in ticker_df.columns:
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MACD'], line=dict(color='cyan', width=1), name='MACD'), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=curr_row, col=1)
            hist = ticker_df['MACD'] - ticker_df['MACD_Signal']
            colors = ['green' if val >= 0 else 'red' for val in hist]
            fig.add_trace(go.Bar(x=ticker_df['Date'], y=hist, marker_color=colors, name='MACD Hist'), row=curr_row, col=1)
            
        elif sp == 'Stochastic' and 'Stoch_K' in ticker_df.columns:
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Stoch_K'], line=dict(color='cyan', width=1), name='%K'), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Stoch_D'], line=dict(color='orange', width=1), name='%D'), row=curr_row, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=curr_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=curr_row, col=1)
            
        elif sp == 'ATR' and 'ATR' in ticker_df.columns:
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['ATR'], line=dict(color='white', width=1), name='ATR'), row=curr_row, col=1)
            
        elif sp == 'OBV' and 'OBV' in ticker_df.columns:
            fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['OBV'], line=dict(color='yellow', width=1), name='OBV'), row=curr_row, col=1)

    # Layout styling
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600 + (200 * num_subplots), # Dynamic height
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_prediction_chart(ticker_df, forecast_df, ticker):
    """
    Generates a comparison chart of Actual vs Predicted prices.
    forecast_df should be filtered for the specific ticker.
    """
    fig = go.Figure()
    
    # 1. Actual Price Line
    fig.add_trace(go.Scatter(
        x=ticker_df['Date'], 
        y=ticker_df['Close'], 
        mode='lines',
        name='Actual Price',
        line=dict(color='cyan', width=2)
    ))
    
    # 2. Predicted Price Scatter/Lines
    # We plot the PREDICTED value on the TARGET date.
    # forecast_df columns: target_date_1d, predicted_1d
    
    if not forecast_df.empty:
        # Ensure dates are datetime
        forecast_df['target_date_1d'] = pd.to_datetime(forecast_df['target_date_1d'])
        forecast_df['predicted_1d'] = pd.to_numeric(forecast_df['predicted_1d'])
        
        # Sort by date and drop duplicates (keep last entry for same target date)
        forecast_df = forecast_df.sort_values(by='target_date_1d', ascending=True)
        forecast_df = forecast_df.drop_duplicates(subset=['target_date_1d'], keep='last')
        
        fig.add_trace(go.Scatter(
            x=forecast_df['target_date_1d'], 
            y=forecast_df['predicted_1d'], 
            mode='markers+lines',
            name='AI Prediction (1-Day)',
            line=dict(color='orange', width=1, dash='dot'),
            marker=dict(symbol='x', size=8, color='orange')
        ))
        
    fig.update_layout(
        title=f'{ticker} AI Accuracy: Actual vs Predicted',
        yaxis_title='Price',
        template="plotly_dark",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_future_forecast_chart(ticker_df, forecast_1d, forecast_5d, ticker):
    """
    Plots the recent actual price alongside the predicted future trajectory. 
    """
    # Get last 60 days
    recent_df = ticker_df.tail(60).copy()
    
    fig = go.Figure()
    
    # 1. Plot Recent Actual Price
    fig.add_trace(go.Scatter(
        x=recent_df['Date'], 
        y=recent_df['Close'], 
        mode='lines',
        name='Actual Price',
        line=dict(color='cyan', width=2)
    ))

    # 2. Setup Future trajectory
    last_row = recent_df.iloc[-1]
    last_date = last_row['Date']
    last_price = last_row['Close']
    
    # Calculate future dates (approximate calendar days for 1d and 5d business trading days)
    date_1d = last_date + timedelta(days=1)
    date_week = last_date + timedelta(days=7)
    
    # The trajectory connects: Current Price -> 1D Forecast -> 5D Forecast
    traj_dates = [last_date, date_1d, date_week]
    traj_prices = [last_price, forecast_1d, forecast_5d]
    
    fig.add_trace(go.Scatter(
        x=traj_dates, 
        y=traj_prices, 
        mode='lines+markers',
        name='AI Fast-Forward',
        line=dict(color='yellow', width=2, dash='dash'),
        marker=dict(symbol='star', size=10, color='yellow')
    ))

    fig.update_layout(
        title=f'{ticker} Extrapolated Trajectory (Next 5 Days)',
        yaxis_title='Price',
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig
