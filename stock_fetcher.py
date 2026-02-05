import yfinance as yf
import pandas as pd
import os

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

            # Handle MultiIndex columns if present (yfinance update)
            # If columns are MultiIndex (Price, Ticker), flatten them or handle properly
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    # In recent yF versions, extracting the scalar value if it's single level
                    if df.columns.nlevels > 1:
                         df.columns = df.columns.get_level_values(0)
                except Exception as e:
                    pass

            # Calculate Moving Averages
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Calculate Bollinger Bands
            # Middle Band = 20-day simple moving average (SMA)
            # Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
            # Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
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
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if all_data:
        print("Concatenating data...")
        final_df = pd.concat(all_data)
        
        # Define relevant columns to keep
        # Ensure 'Date' is present (it was the index)
        # Standard yfinance columns: Date, Open, High, Low, Close, Adj Close, Volume
        
        # Reorder columns slightly
        columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
                           'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'RSI']
        
        # Filter only existing columns just in case
        existing_cols = [c for c in columns_to_keep if c in final_df.columns]
        final_df = final_df[existing_cols]
        
        # Save to CSV
        output_file = 'stocks_data.csv'
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        
        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved stock data to {output_path}")
        print(f"Total records: {len(final_df)}")
        print(final_df.head())
    else:
        print("No stock data collected.")

if __name__ == "__main__":
    fetch_stock_data()
