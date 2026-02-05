# How to Deploy Your Stock App Online 🚀

To share your dashboard with the world, the easiest (and free) way is **Streamlit Community Cloud**.

## Prerequisites
1.  **A GitHub Account**: ([Sign up here](https://github.com/))
2.  **This Code**: You likely need to upload your `StocksAnalysis` folder to a GitHub repository.

## Step 1: Upload Code to GitHub
1.  Create a **New Repository** on GitHub (e.g., `stock-analysis-dashboard`).
2.  Upload the following files to it:
    *   `stock_app.py`
    *   `stock_fetcher.py`, `stock_visualizer.py`, `stock_forecaster.py`, `stock_backtester.py`
    *   `stocks_data.csv` (Optional, the app can refetch, but good to have)
    *   `external_index.csv` (Important! Your AI Brain needs this)
    *   `requirements.txt` (Crucial! Tells the cloud what libraries to install)

## Step 2: Connect to Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"Sign up with GitHub"**.
3.  Click **"New app"**.
4.  Select your new repository (`stock-analysis-dashboard`) and branch (`main`).
5.  **Main file path**: Enter `stock_app.py` (or `StocksAnalysis/stock_app.py` if it's inside a folder).
6.  Click **"Deploy!"**.

## Step 3: Watch it Build
Streamlit will read your `requirements.txt`, install `yfinance` and `scikit-learn`, and launch your app. In about 2-3 minutes, you'll have a public URL (like `https://stock-dashboard.streamlit.app`) to share with anyone!

## Troubleshooting
- **"Module not found"**: Ensure `requirements.txt` is in the same folder as `stock_app.py` or at the root.
- **"File not found"**: If you put files in a folder, make sure your code uses relative paths correctly (The current code already uses `os.path.join(current_dir, ...)` which is cloud-friendly!).
