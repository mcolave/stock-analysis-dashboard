import yfinance as yf
import pandas as pd
import os

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
    
    # Add Ticker column
    df['Ticker'] = ticker
    
    # Reset index to transform Date index into a column
    df = df.reset_index()
    
    # Define relevant columns to keep
    columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI']
    
    # Filter only existing columns
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    return df[existing_cols]

def get_data_path():
    return os.path.join(os.path.dirname(__file__), 'stocks_data.csv')

def add_ticker(ticker_symbol):
    """
    Adds a new ticker to the existing dataset.
    Returns string message (success or error).
    """
    ticker_symbol = ticker_symbol.upper().strip()
    file_path = get_data_path()
    
    if not os.path.exists(file_path):
        return "Error: Data file not found. Run full fetch first."
        
    # Check if already exists
    try:
        existing_df = pd.read_csv(file_path)
        if ticker_symbol in existing_df['Ticker'].unique():
            return f"Ticker {ticker_symbol} is already in the database."
    except Exception as e:
        return f"Error reading database: {e}"
        
    print(f"Fetching data for new ticker: {ticker_symbol}...")
    try:
        df = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if df.empty:
            return f"No data found for {ticker_symbol}. Check spelling."
            
        processed_df = process_stock_data(df, ticker_symbol)
        
        # Append to CSV
        # We read, append, and save back. 
        # (For very large files, append mode 'a' is better, but for <50MB this is safe and cleaner for deduplication if needed)
        updated_df = pd.concat([existing_df, processed_df])
        updated_df.to_csv(file_path, index=False)
        
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
        
        output_path = get_data_path()
        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved stock data to {output_path}")
        print(f"Total records: {len(final_df)}")
        print(final_df.head())
    else:
        print("No stock data collected.")

if __name__ == "__main__":
    fetch_stock_data()
