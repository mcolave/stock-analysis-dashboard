import schedule
import time
import pandas as pd
import stock_fetcher
import stock_forecaster
import db_manager
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(filename='auto_runner.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def job():
    print("Starting Auto-Runner Job...")
    logging.info("Starting Auto-Runner Job...")
    
    # 1. Fetch Latest Data
    try:
        print("Fetching latest stock data...")
        stock_fetcher.fetch_stock_data()
        logging.info("Stock data fetched successfully.")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        print(f"Error fetching data: {e}")
        return # Stop if fetch fails
        
    # 2. Load Data from DB
    df = db_manager.load_price_data()
    if df is None or df.empty:
        logging.error("No data found in DB.")
        return

    # 3. Run Forecasts for ALL Tickers
    tickers = df['Ticker'].unique()
    print(f"Running forecasts for {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        try:
            print(f"Forecasting {ticker}...")
            res = stock_forecaster.run_forecast(df, ticker)
            
            if "error" in res:
                logging.warning(f"Skipping {ticker}: {res['error']}")
                continue
                
            # Prepare data for saving
            # We need to calculate target dates based on latest date in data
            # Note: forecast returns current_price and forecasts, but not dates explicitly in dict 
            # We need to reconstruct them logic from app.
            
            ticker_df = df[df['Ticker'] == ticker]
            latest_date = ticker_df['Date'].max()
            
            target_1d = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            target_1w = (latest_date + timedelta(days=5)).strftime('%Y-%m-%d')
            
            forecast_data = {
                'date_logged': datetime.now().strftime('%Y-%m-%d'),
                'ticker': ticker,
                'target_1d': target_1d,
                'pred_1d': round(res['forecast_1day'], 2),
                'target_1w': target_1w,
                'pred_1w': round(res['forecast_5days'], 2),
                'model': res['best_model_1day'],
                'adj_info': "Auto-Run (No Manual Events)"
            }
            
            # Save to DB
            db_manager.save_forecast(forecast_data)
            logging.info(f"Saved forecast for {ticker}.")
            
        except Exception as e:
            logging.error(f"Error forecasting {ticker}: {e}")

    print("Auto-Runner Job Complete.")
    logging.info("Auto-Runner Job Complete.")

if __name__ == "__main__":
    # If run directly, run once immediately
    job()
    
    # Optional: Keep running if intended as a service
    # print("Scheduler started. Waiting for 10:00 every day...")
    # schedule.every().day.at("10:00").do(job)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
