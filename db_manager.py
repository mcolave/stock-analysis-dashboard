import sqlite3
import pandas as pd
import os

DB_NAME = "stocks.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_logged TEXT,
            ticker TEXT,
            target_date_1d TEXT,
            predicted_1d REAL,
            target_date_1w TEXT,
            predicted_1w REAL,
            model_used TEXT,
            adjustment_info TEXT,
            actual_1d REAL,
            actual_1w REAL,
            error_1d_pct REAL,
            error_1w_pct REAL
        )
    ''')
    conn.commit()
    conn.close()

def init_prices_table():
    conn = get_db_connection()
    c = conn.cursor()
    # Composite PK on (date, ticker) ensures no duplicates
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            sma_50 REAL,
            sma_200 REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            atr REAL,
            obv REAL,
            PRIMARY KEY (date, ticker)
        )
    ''')
    conn.commit()
    conn.close()

# Update init_db to also init prices
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_logged TEXT,
            ticker TEXT,
            target_date_1d TEXT,
            predicted_1d REAL,
            target_date_1w TEXT,
            predicted_1w REAL,
            model_used TEXT,
            adjustment_info TEXT,
            actual_1d REAL,
            actual_1w REAL,
            error_1d_pct REAL,
            error_1w_pct REAL
        )
    ''')
    conn.commit()
    conn.close()
    
    # Also init prices
    init_prices_table()

def save_forecast(data):
    """
    Saves a forecast dictionary to the database.
    Expected keys in data: date_logged, ticker, target_1d, pred_1d, target_1w, pred_1w, model, adj_info
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO forecasts (
            date_logged, ticker, target_date_1d, predicted_1d, 
            target_date_1w, predicted_1w, model_used, adjustment_info
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['date_logged'],
        data['ticker'],
        data['target_1d'],
        data['pred_1d'],
        data['target_1w'],
        data['pred_1w'],
        data['model'],
        data['adj_info']
    ))
    conn.commit()
    conn.close()

def get_all_forecasts():
    """
    Returns all forecasts as a pandas DataFrame.
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM forecasts", conn)
    conn.close()
    return df

def update_actual_price_1d(row_id, actual_price, error_pct):
    """
    Updates the actual 1D price and error percentage for a specific forecast record.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE forecasts 
        SET actual_1d = ?, error_1d_pct = ?
        WHERE id = ?
    ''', (actual_price, error_pct, row_id))
    conn.commit()
    conn.close()

def save_price_data(df):
    """
    Saves a dataframe of stock prices to the database.
    Expects DataFrame with columns matching the table schema.
    Uses INSERT OR REPLACE to handle updates/duplicates efficiently.
    """
    conn = get_db_connection()
    
    # Ensure column names map correctly to DB (lowercase)
    # The DF likely has Title Case (Open, Close), we need to map or rename
    # Simple strategy: Rename cols to lowercase
    df_clean = df.copy()
    df_clean.columns = [c.lower() for c in df_clean.columns]
    
    # We need to manually iterate or use to_sql with if_exists='append' but we want to handle duplicates.
    # Pandas to_sql doesn't support UPSERT easily in sqlite.
    # So we used a looped INSERT OR REPLACE approach for safety, or we delete old records for these tickers and insert new.
    # "Delete and Replace" is risky if we only fetch partial data.
    # "Insert or Replace" is best.
    
    c = conn.cursor()
    
    # Prepare list of tuples
    # Ensure all required columns exist, fill missing with None
    required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                     'sma_50', 'sma_200', 'bb_upper', 'bb_middle', 'bb_lower', 
                     'rsi', 'macd', 'macd_signal', 'atr', 'obv']
    
    for col in required_cols:
        if col not in df_clean.columns:
            df_clean[col] = None
            
    # Convert date to string if needed
    if pd.api.types.is_datetime64_any_dtype(df_clean['date']):
        df_clean['date'] = df_clean['date'].dt.strftime('%Y-%m-%d')
        
    records = df_clean[required_cols].to_dict('records')
    
    c.executemany('''
        INSERT OR REPLACE INTO stock_prices (
            date, ticker, open, high, low, close, volume, 
            sma_50, sma_200, bb_upper, bb_middle, bb_lower, 
            rsi, macd, macd_signal, atr, obv
        ) VALUES (
            :date, :ticker, :open, :high, :low, :close, :volume,
            :sma_50, :sma_200, :bb_upper, :bb_middle, :bb_lower,
            :rsi, :macd, :macd_signal, :atr, :obv
        )
    ''', records)
    
    conn.commit()
    conn.close()

def load_price_data(ticker=None):
    """
    Loads price data. If ticker is provided, filters by ticker.
    Returns DataFrame with Title Case columns to match app expectation.
    """
    conn = get_db_connection()
    
    query = "SELECT * FROM stock_prices"
    params = ()
    
    if ticker:
        query += " WHERE ticker = ?"
        params = (ticker,)
        
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return None
        
    # Rename columns back to Title Case for compatibility with existing app logic
    # Mapping:
    col_map = {
        'date': 'Date', 'ticker': 'Ticker', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        'sma_50': 'SMA_50', 'sma_200': 'SMA_200', 'bb_upper': 'BB_Upper', 'bb_middle': 'BB_Middle', 'bb_lower': 'BB_Lower',
        'rsi': 'RSI', 'macd': 'MACD', 'macd_signal': 'MACD_Signal', 'atr': 'ATR', 'obv': 'OBV'
    }
    df = df.rename(columns=col_map)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df
