import pandas as pd
import numpy as np
import db_manager
import stock_forecaster
from datetime import datetime, timedelta
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backfill():
    print("Starting Backfill Process (Last 30 Days)...")
    
    # 1. Load ALL Data
    df = db_manager.load_price_data()
    if df is None or df.empty:
        print("No data found in DB.")
        return

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    tickers = df['Ticker'].unique()
    
    # 2. Define Date Range (Yesterday back to 10 days ago for speed)
    # We want to fill the "gap" of history.
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=10)
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    print(f"Backfilling {len(date_range)} days (from {start_date.date()} to {end_date.date()}) for {len(tickers)} tickers.")
    
    count = 0
    conn = db_manager.get_db_connection() # Keep one connection open
    
    for current_date in date_range:
        current_date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing {current_date_str}...")
        
        for ticker in tickers:
            # 3. Create Historical View
            historical_view = df[(df['Ticker'] == ticker) & (df['Date'] <= current_date)].copy()
            
            if len(historical_view) < 60:
                continue
                
            try:
                # 4. Run Forecast
                res = stock_forecaster.run_forecast(historical_view, ticker)
                
                if "error" in res:
                    continue
                    
                # 5. Prepare Record
                target_1d_date = current_date + timedelta(days=1)
                target_1w_date = current_date + timedelta(days=5)
                
                # Check for Actuals
                actual_1d = None
                error_1d_pct = None
                
                full_ticker_data = df[df['Ticker'] == ticker]
                actual_row = full_ticker_data[full_ticker_data['Date'] == target_1d_date]
                
                if not actual_row.empty:
                    actual_1d = float(actual_row['Close'].iloc[0])
                    pred = float(res['forecast_1day'])
                    error_1d_pct = abs(actual_1d - pred) / actual_1d * 100
                    
                # 6. Save directly
                c = conn.cursor()
                c.execute('''
                    INSERT INTO forecasts (
                        date_logged, ticker, target_date_1d, predicted_1d, 
                        target_date_1w, predicted_1w, model_used, adjustment_info,
                        actual_1d, error_1d_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_date_str,
                    ticker,
                    target_1d_date.strftime('%Y-%m-%d'),
                    round(res['forecast_1day'], 2),
                    target_1w_date.strftime('%Y-%m-%d'),
                    round(res['forecast_5days'], 2),
                    res['best_model_1day'],
                    "Backfilled AI Data",
                    actual_1d,
                    error_1d_pct
                ))
                conn.commit()
                count += 1
                
            except Exception as e:
                logging.error(f"Error on {ticker} {current_date_str}: {e}")
                
    conn.close()
    print(f"Backfill Complete. Generated {count} historical predictions.")
                
    print(f"Backfill Complete. Generated {count} historical predictions.")

if __name__ == "__main__":
    backfill()
