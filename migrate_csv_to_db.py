import pandas as pd
import db_manager
import os

def migrate():
    print("Starting migration from CSV to SQLite...")
    
    csv_path = 'stocks_data.csv'
    if not os.path.exists(csv_path):
        print("No CSV file found to migrate.")
        return

    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Read {len(df)} rows from CSV.")
        
        # Init DB (creates table if missing)
        db_manager.init_db()
        
        # Save to DB
        db_manager.save_price_data(df)
        print("Data successfully saved to Database!")
        
        # Optional: Rename CSV to indicate it's backed up
        # os.rename(csv_path, 'stocks_data_backup.csv')
        # print("Renamed stocks_data.csv to stocks_data_backup.csv")
        
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
