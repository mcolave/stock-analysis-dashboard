import yfinance as yf
import pandas as pd
import numpy as np
import os
import db_manager

def process_stock_data(df, ticker):
    """
    Calculates indicators and prepares dataframe for storage.
    """
    # Handle MultiIndex columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    # Calculate Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Calculate RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- NEW INDICATORS ---
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR (14)
    # TR = Max(|High-Low|, |High-PrevClose|, |Low-PrevClose|)
    df['PrevClose'] = df['Close'].shift(1)
    df['TR1'] = df['High'] - df['Low']
    df['TR2'] = abs(df['High'] - df['PrevClose'])
    df['TR3'] = abs(df['Low'] - df['PrevClose'])
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # OBV (On-Balance Volume)
    # If Close > PrevClose, +Volume. If Close < PrevClose, -Volume.
    df['OBV_Multiplier'] = np.where(df['Close'] > df['PrevClose'], 1, -1)
    df['OBV_Multiplier'] = np.where(df['Close'] == df['PrevClose'], 0, df['OBV_Multiplier'])
    df['OBV'] = (df['Volume'] * df['OBV_Multiplier']).cumsum()

    
    # Add Ticker column
    df['Ticker'] = ticker
    
    # Reset index to transform Date index into a column
    df = df.reset_index()
    
    # Define relevant columns to keep
    columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI',
                        'MACD', 'MACD_Signal', 'ATR', 'OBV']
    
    # Filter only existing columns
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    return df[existing_cols]



def add_ticker(ticker_symbol):
    """
    Adds a new ticker to the existing dataset.
    Returns string message (success or error).
    """
    ticker_symbol = ticker_symbol.upper().strip()
    ticker_symbol = ticker_symbol.upper().strip()
    
    # Check if already exists in DB
    try:
        existing_df = db_manager.load_price_data(ticker_symbol)
        if existing_df is not None and not existing_df.empty:
            return f"Ticker {ticker_symbol} is already in the database."
    except Exception as e:
        return f"Error reading database: {e}"
        
    print(f"Fetching data for new ticker: {ticker_symbol}...")
    try:
        df = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if df.empty:
            return f"No data found for {ticker_symbol}. Check spelling."
            
        processed_df = process_stock_data(df, ticker_symbol)
        
        processed_df = process_stock_data(df, ticker_symbol)
        
        # Save to DB
        db_manager.save_price_data(processed_df)
        
        return f"Successfully added {ticker_symbol}!"
        
    except Exception as e:
        return f"Error adding {ticker_symbol}: {e}"

def fetch_stock_data():
    # List of tickers to track - starting with major tech stocks and STX
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'STX']
    all_data = []

    print("Starting stock data download...")

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Download last 2 years of daily data
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            if df.empty:
                print(f"No data found for {ticker}")
                continue

            processed_df = process_stock_data(df, ticker)
            all_data.append(processed_df)
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if all_data:
        print("Concatenating data...")
        final_df = pd.concat(all_data)
        
    if all_data:
        print("Concatenating data...")
        final_df = pd.concat(all_data)
        
        db_manager.save_price_data(final_df)
        print(f"Successfully saved stock data to Database")
        print(f"Total records: {len(final_df)}")
        print(final_df.head())
    else:
        print("No stock data collected.")

if __name__ == "__main__":
    db_manager.init_db() # Ensure DB exists
    fetch_stock_data()
