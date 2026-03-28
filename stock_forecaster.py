import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(model_class, model_kwargs, X, y):
    """
    Trains across TimeSeries splits to get a robust MAE.
    Returns the average MAE, a model fitted on all provided data, and the scaler.
    """
    tscv = TimeSeriesSplit(n_splits=4)
    maes = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # X is a dataframe, output is numpy array
    
    # We must reset index of y so iloc alignment works perfectly with enumeration
    y_vals = y.values
    
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_vals[train_index], y_vals[test_index]
        
        # Instantiate fresh model for this fold
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        
    avg_mae = np.mean(maes)
    
    # Train final model on ALL data in this set
    final_model = model_class(**model_kwargs)
    final_model.fit(X_scaled, y_vals)
    
    return avg_mae, final_model, scaler

def run_forecast(df, ticker):
    """
    Runs multiple models (RF, GB, LR) and selects the best one.
    Returns dict with predictions, metrics, and model comparison.
    """
    if 'Ticker' in df.columns:
        ticker_df = df[df['Ticker'] == ticker].copy()
    else:
        ticker_df = df.copy()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV', 'VIX_Close', 'Oil_Close', 'Gold_Close', 'USD_Close']
    
    # Check if we have all features
    missing_cols = [col for col in features if col not in ticker_df.columns]
    if missing_cols:
        return {"error": f"Missing data columns: {missing_cols}. Please click 'Refetch ALL Data' in the sidebar to generate them."}
    
    # Drop rows where features are NaN
    ticker_df = ticker_df.dropna(subset=features)
    
    if len(ticker_df) < 50:
        return {"error": "Not enough data to model."}

    # Create Targets
    ticker_df['Target_NextDay'] = ticker_df['Close'].shift(-1)
    ticker_df['Target_NextWeek'] = ticker_df['Close'].shift(-5)
    
    # Prepare training data
    data_for_next_day = ticker_df.dropna(subset=['Target_NextDay'])
    data_for_next_week = ticker_df.dropna(subset=['Target_NextWeek'])
    
    results = {}
    
    # Define Models to Try
    models_config = {
        'Random Forest': (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
        'Gradient Boosting': (GradientBoostingRegressor, {'n_estimators': 100, 'random_state': 42}),
        'Linear Regression': (LinearRegression, {}),
        'Ridge Regression': (Ridge, {'alpha': 1.0}),
        'SVR (Support Vector)': (SVR, {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'})
    }
    
    # --- Next Day Model Selection ---
    X_day = data_for_next_day[features]
    y_day = data_for_next_day['Target_NextDay']
    
    best_mae_day = float('inf')
    best_model_name_day = None
    best_model_day = None
    best_scaler_day = None
    
    day_metrics = {}
    
    for name, (m_class, m_kwargs) in models_config.items():
        mae, final_m, scaler = train_and_evaluate(m_class, m_kwargs, X_day, y_day)
        day_metrics[name] = mae
        if mae < best_mae_day:
            best_mae_day = mae
            best_model_name_day = name
            best_model_day = final_m
            best_scaler_day = scaler
    
    results['mae_1day'] = best_mae_day
    results['best_model_1day'] = best_model_name_day
    results['comparison_1day'] = day_metrics
    
    # --- Next Week Model Selection ---
    X_week = data_for_next_week[features]
    y_week = data_for_next_week['Target_NextWeek']
    
    best_mae_week = float('inf')
    best_model_name_week = None
    best_model_week = None
    best_scaler_week = None
    
    week_metrics = {}
    
    for name, (m_class, m_kwargs) in models_config.items():
        mae, final_m, scaler = train_and_evaluate(m_class, m_kwargs, X_week, y_week)
        week_metrics[name] = mae
        if mae < best_mae_week:
            best_mae_week = mae
            best_model_name_week = name
            best_model_week = final_m
            best_scaler_week = scaler
    
    results['mae_5days'] = best_mae_week
    results['best_model_5days'] = best_model_name_week
    results['comparison_5days'] = week_metrics
    
    # --- Predict Future ---
    latest_data = ticker_df.iloc[[-1]][features]
    results['current_price'] = latest_data['Close'].values[0]
    
    # MUST scale the latest_data using the respective scalers!
    latest_scaled_day = best_scaler_day.transform(latest_data)
    latest_scaled_week = best_scaler_week.transform(latest_data)
    
    results['forecast_1day'] = best_model_day.predict(latest_scaled_day)[0]
    results['forecast_5days'] = best_model_week.predict(latest_scaled_week)[0]
    
    # We will also output current macro logic for the HUD
    results['latest_macro'] = {
        'VIX': latest_data['VIX_Close'].values[0],
        'Oil': latest_data['Oil_Close'].values[0],
        'Gold': latest_data['Gold_Close'].values[0],
        'USD': latest_data['USD_Close'].values[0]
    }
    
    return results
