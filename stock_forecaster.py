import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(model, X, y):
    """
    Helper to train a model and return MAE on test set.
    """
    # Validation Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae, model

def run_forecast(df, ticker):
    """
    Runs multiple models (RF, GB, LR) and selects the best one.
    Returns dict with predictions, metrics, and model comparison.
    """
    if 'Ticker' in df.columns:
        ticker_df = df[df['Ticker'] == ticker].copy()
    else:
        ticker_df = df.copy()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV']
    
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
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    # --- Next Day Model Selection ---
    X_day = data_for_next_day[features]
    y_day = data_for_next_day['Target_NextDay']
    
    best_mae_day = float('inf')
    best_model_name_day = None
    best_model_day = None
    
    day_metrics = {}
    
    for name, model_inst in models.items():
        # Clone model instance (re-init) to be safe or just fit directly as they are fresh
        mae, _ = train_and_evaluate(model_inst, X_day, y_day)
        day_metrics[name] = mae
        if mae < best_mae_day:
            best_mae_day = mae
            best_model_name_day = name
            best_model_day = model_inst # This is the fitted instance on train set
            
    # Re-train BEST model on FULL data
    best_model_day.fit(X_day, y_day)
    
    results['mae_1day'] = best_mae_day
    results['best_model_1day'] = best_model_name_day
    results['comparison_1day'] = day_metrics
    
    # --- Next Week Model Selection ---
    X_week = data_for_next_week[features]
    y_week = data_for_next_week['Target_NextWeek']
    
    best_mae_week = float('inf')
    best_model_name_week = None
    best_model_week = None
    
    week_metrics = {}
    
    for name, model_inst in models.items():
        # Re-instantiate for week target to avoid leakage/state issues
        if name == 'Random Forest': model_inst = RandomForestRegressor(n_estimators=100, random_state=42)
        elif name == 'Gradient Boosting': model_inst = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else: model_inst = LinearRegression()
            
        mae, _ = train_and_evaluate(model_inst, X_week, y_week)
        week_metrics[name] = mae
        if mae < best_mae_week:
            best_mae_week = mae
            best_model_name_week = name
            best_model_week = model_inst
            
    best_model_week.fit(X_week, y_week)
    
    results['mae_5days'] = best_mae_week
    results['best_model_5days'] = best_model_name_week
    results['comparison_5days'] = week_metrics
    
    # --- Predict Future ---
    latest_data = ticker_df.iloc[[-1]][features]
    results['current_price'] = latest_data['Close'].values[0]
    
    results['forecast_1day'] = best_model_day.predict(latest_data)[0]
    results['forecast_5days'] = best_model_week.predict(latest_data)[0]
    
    return results
