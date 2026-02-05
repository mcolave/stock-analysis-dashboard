import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

def run_backtest(df, ticker):
    """
    Runs a backtest simulation for a given ticker.
    Returns metrics dict and a Plotly figure.
    """
    if 'Ticker' in df.columns:
        ticker_df = df[df['Ticker'] == ticker].copy()
    else:
        ticker_df = df.copy()
        
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI']
    ticker_df = ticker_df.dropna(subset=features)
    
    # Target: Next Day Close
    ticker_df['Next_Close'] = ticker_df['Close'].shift(-1)
    ticker_df = ticker_df.dropna(subset=['Next_Close'])
    
    X = ticker_df[features]
    y = ticker_df['Next_Close']
    dates = ticker_df['Date']
    
    # Train/Test Split
    split_idx = int(len(ticker_df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    test_dates = dates.iloc[split_idx:]
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Simulation Prep
    results = X_test.copy()
    results['Date'] = test_dates
    results['Actual_Next_Close'] = y_test
    results['Predicted_Next_Close'] = predictions
    
    # Align Next Day Open
    ticker_df['Next_Open'] = ticker_df['Open'].shift(-1)
    next_open_test = ticker_df['Next_Open'].iloc[split_idx:]
    results['Actual_Next_Open'] = next_open_test
    
    results = results.dropna()
    
    # Strategy Logic: Buy Open if Predicted Price > Open
    results['Signal'] = np.where(results['Predicted_Next_Close'] > results['Actual_Next_Open'], 1, 0)
    
    # Returns
    results['Strategy_Return'] = results['Signal'] * ((results['Actual_Next_Close'] - results['Actual_Next_Open']) / results['Actual_Next_Open'])
    results['BuyHold_Return'] = (results['Actual_Next_Close'] - results['Actual_Next_Open']) / results['Actual_Next_Open']
    
    # Cumulative Equity
    results['Strategy_Equity'] = (1 + results['Strategy_Return']).cumprod()
    results['BuyHold_Equity'] = (1 + results['BuyHold_Return']).cumprod()
    
    # Metrics
    total_trades = results['Signal'].sum()
    win_trades = results[(results['Signal'] == 1) & (results['Strategy_Return'] > 0)]
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    
    metrics = {
        'trades': int(total_trades),
        'win_rate': win_rate,
        'strategy_return': results['Strategy_Equity'].iloc[-1] - 1,
        'buy_hold_return': results['BuyHold_Equity'].iloc[-1] - 1
    }
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['Date'], y=results['Strategy_Equity'], mode='lines', name='Model Strategy', line=dict(color='#00CC96')))
    fig.add_trace(go.Scatter(x=results['Date'], y=results['BuyHold_Equity'], mode='lines', name='Buy & Hold', line=dict(color='#636EFA')))
    
    fig.update_layout(
        title=f'{ticker} Backtest Equity Curve',
        yaxis_title='Growth (1.0 = Start)',
        template='plotly_dark',
        height=500
    )
    
    return metrics, fig
