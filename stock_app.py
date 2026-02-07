import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import stock_visualizer
import stock_forecaster
import stock_backtester
import stock_fetcher
import db_manager
import stock_analyst

# Initialize DB on startup
db_manager.init_db()

# Page Config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide", page_icon="📈")

# Function to load data
@st.cache_data
def load_data():
    return db_manager.load_price_data()

# Sidebar
st.sidebar.title("Configuration")

# Data Management
with st.sidebar.expander("Manage Data", expanded=False):
    if st.button("Refetch ALL Data (Slow)"):
        with st.spinner("Fetching all stock data..."):
            stock_fetcher.fetch_stock_data()
            st.cache_data.clear()
            st.success("All data updated!")
            st.rerun()

    st.write("---")
    new_ticker = st.text_input("Add New Ticker:", placeholder="e.g. AMD")
    if st.button("Add Ticker"):
        if new_ticker:
            with st.spinner(f"Fetching data for {new_ticker}..."):
                msg = stock_fetcher.add_ticker(new_ticker)
                
                if "Success" in msg:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.warning("Please enter a ticker symbol.")

df = load_data()

if df is None:
    st.error("No data found. Please add a ticker or click 'Refetch Data'.")
    st.stop()

# Ticker Selection
if 'Ticker' in df.columns:
    tickers = df['Ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers, index=0)
    
    # Filter Data
    ticker_df = df[df['Ticker'] == selected_ticker].copy()
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
    latest_price = ticker_df['Close'].iloc[-1]
    latest_date = ticker_df['Date'].iloc[-1].strftime('%Y-%m-%d')
else:
    st.error("Invalid CSV format. Expected 'Ticker' column.")
    st.stop()

# Header
st.title(f"{selected_ticker} Analysis Dashboard")
st.markdown(f"**Current Price:** ${latest_price:.2f} &nbsp;|&nbsp; **Date:** {latest_date}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart Analysis", "🔮 Forecasting", "🔙 Backtesting", "✅ Accuracy Tracker"])

# Tab 1: Visualization
with tab1:
    st.subheader("Technical Analysis Chart")
    with st.expander("ℹ️ Need help? Explanation for Dummies"):
        st.markdown("""
        **What am I looking at?**
        *   **Bollinger Bands (Gray Shaded)**: Shows volatility. Price usually stays inside these bands. If it touches the top, it might be expensive (Overbought). If it touches the bottom, it might be cheap (Oversold).
        *   **RSI (Purple Line)**: Momentum. Above 70 = Overbought. Below 30 = Oversold.
        *   **MACD (Bottom Panel)**: Trend finder.
            *   *bars (Green/Red)*: Momentum strength. Green = Bullish (Up), Red = Bearish (Down).
            *   *Lines (Cyan/Orange)*: When Cyan crosses above Orange, it's often a Buy signal.
        """)
    fig = stock_visualizer.create_chart(df, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Plain English Analysis
    with st.expander("📝 Instant Analysis (Plain English)", expanded=True):
        analysis_text = stock_analyst.generate_summary(ticker_df, selected_ticker)
        st.markdown(analysis_text)

# Tab 2: Forecast
with tab2:
    st.subheader("AI Price Prediction")
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **The Battle of the Models:**
        We don't just use one brain; we use three!
        
        1.  **Random Forest**: A committee of decision trees.
        2.  **Gradient Boosting**: A team of trees that learn from each other's mistakes (usually the smartest).
        3.  **Linear Regression**: The simple baseline (drawing a straight line).
        
        We let them all fight (train) on past data. The one with the lowest error (MAE) wins and gets to predict the future!
        """)
    
    st.write("---")
    st.write("---")
    st.markdown("### 🤖 ai Event Simulator (Data-Driven)")
    st.caption("Select upcoming events. The AI will apply historical impact factors (Short Term vs Medium Term) from `external_index.csv`.")
    
    # Load External Index
    index_path = os.path.join(current_dir, 'external_index.csv')
    if os.path.exists(index_path):
        external_df = pd.read_csv(index_path)
        
        # Display Format: "Event Name (1D: -2%, 1W: -5%)"
        external_df['Display'] = external_df.apply(
            lambda x: f"{x['Event']} (1D: {x['Impact_1D']}%, 1W: {x['Impact_1W']}%)", axis=1
        )
        event_options = external_df['Display'].tolist()
        
        selected_events = st.multiselect("Select Active Scenarios:", event_options)
        
        # Calculate Adjustment
        adj_factor_1d = 0.0
        adj_factor_1w = 0.0
        active_impacts = []
        
        if selected_events:
            for item in selected_events:
                # Lookup
                row = external_df[external_df['Display'] == item].iloc[0]
                
                i_1d = row['Impact_1D']
                i_1w = row['Impact_1W']
                
                adj_factor_1d += i_1d
                adj_factor_1w += i_1w
                
                active_impacts.append(f"{row['Event']}")
            
            st.info(f"**Applied Events:** {', '.join(active_impacts)}")
            col_adj1, col_adj2 = st.columns(2)
            col_adj1.markdown(f"**Total 1-Day Impact:** `{adj_factor_1d:+.1f}%`")
            col_adj2.markdown(f"**Total 1-Week Impact:** `{adj_factor_1w:+.1f}%`")
        else:
            st.info("No external scenarios selected. Using pure technical forecast.")
            
    else:
        st.warning("⚠️ `external_index.csv` not found. Please create it to use the AI Simulator.")
        adj_factor_1d = 0.0
        adj_factor_1w = 0.0

    st.write("---")

    if st.button("Run Forecast (Train 3 Models)"):
        with st.spinner("Training Random Forest, Gradient Boosting, and Linear Regression..."):
            res = stock_forecaster.run_forecast(df, selected_ticker)
            
            if "error" in res:
                st.error(res["error"])
            else:
                # Apply Scenarios
                forecast_1d_adj = res['forecast_1day'] * (1 + adj_factor_1d/100)
                forecast_5d_adj = res['forecast_5days'] * (1 + adj_factor_1w/100)
                
                # Store in session state for saving
                st.session_state['last_forecast'] = {
                    'ticker': selected_ticker,
                    'date_logged': datetime.now().strftime('%Y-%m-%d'),
                    'target_1d': (pd.to_datetime(latest_date) + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'pred_1d': round(forecast_1d_adj, 2),
                    'target_1w': (pd.to_datetime(latest_date) + timedelta(days=5)).strftime('%Y-%m-%d'),
                    'pred_1w': round(forecast_5d_adj, 2),
                    'model': res['best_model_1day'], # Simplification: tracking 1D best model
                    'adj_info': f"1D:{adj_factor_1d}%|1W:{adj_factor_1w}%"
                }

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Next Day Forecast")
                    
                    if adj_factor_1d == 0:
                         st.metric("Price", f"${res['forecast_1day']:.2f}", 
                              delta=f"{res['forecast_1day'] - res['current_price']:.2f}")
                    else:
                        st.metric("Adjusted Price", f"${forecast_1d_adj:.2f}", 
                              delta=f"{forecast_1d_adj - res['current_price']:.2f}")
                        st.caption(f"Raw Prediction: ${res['forecast_1day']:.2f} | Adj: {adj_factor_1d:+.1f}%")
                        
                    st.success(f"🏆 Best Model: **{res['best_model_1day']}**")
                    st.markdown("**Model Accuracy Competition (Lower Error IS Better):**")
                    st.dataframe(pd.DataFrame.from_dict(res['comparison_1day'], orient='index', columns=['MAE Error']).sort_values('MAE Error'))
                    
                with col2:
                    st.markdown("#### Next Week Forecast (5 Days)")
                    
                    if adj_factor_1w == 0:
                        st.metric("Price", f"${res['forecast_5days']:.2f}",
                              delta=f"{res['forecast_5days'] - res['current_price']:.2f}")
                    else:
                        st.metric("Adjusted Price", f"${forecast_5d_adj:.2f}",
                              delta=f"{forecast_5d_adj - res['current_price']:.2f}")
                        st.caption(f"Raw Prediction: ${res['forecast_5days']:.2f} | Adj: {adj_factor_1w:+.1f}%")
                    
                    st.success(f"🏆 Best Model: **{res['best_model_5days']}**")
                    st.markdown("**Model Accuracy Competition (Lower Error IS Better):**")
                    st.dataframe(pd.DataFrame.from_dict(res['comparison_5days'], orient='index', columns=['MAE Error']).sort_values('MAE Error'))

    # Save Button Section
    if 'last_forecast' in st.session_state and st.session_state['last_forecast']['ticker'] == selected_ticker:
        st.write("---")
        if st.button("💾 Save Forecast to History"):
            lf = st.session_state['last_forecast']
            
            # Save to DB
            db_manager.save_forecast(lf)
            
            st.success("Forecast saved! Check the 'Accuracy Tracker' tab.")

# Tab 3: Backtest
with tab3:
    st.subheader("Strategy Backtest (Last 20% Data)")
    st.info("Strategy: Buy Open if Model Predicts Higher Close. Sell Close.")
    
    with st.expander("ℹ️ Understanding these numbers"):
        st.markdown("""
        **Did the robot make money?**
        We travelled back in time to the last few months and let the AI trade using its own predictions.
        
        *   **Win Rate**: The percentage of trades that made a profit. > 50% is decent.
        *   **Strategy Return**: How much money the AI made (AFTER paying a 0.1% fee per trade).
        *   **Buy & Hold Return**: How much you would have made if you just bought the stock and did nothing.
        *   **Sharpe Ratio**: **Risk vs. Reward.**
            *   *Above 1.0*: Good (More return for the risk).
            *   *Below 1.0*: Bad (Too much risk for little return).
        *   **Max Drawdown**: **The "Stomach Ache" Factor.** The worst drop your account would have suffered from the peak.
            *   e.g. -20% means at one point, you lost 20% of your account value. Smaller is better!
        *   **Equity Curve**: Green line = AI. Blue line = Lazy Buy & Hold.
        """)
    
    if st.button("Run Backtest"):
        with st.spinner("Simulating trades..."):
            metrics, fig = stock_backtester.run_backtest(df, selected_ticker)
            
            # Metrics
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            m_col1.metric("Total Trades", metrics['trades'])
            m_col2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
            m_col3.metric("Strategy Return", f"{metrics['strategy_return']:.2%}")
            m_col4.metric("Buy & Hold", f"{metrics['buy_hold_return']:.2%}")
            m_col5.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            m_col6.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            
            # Chart
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: Accuracy Tracker
with tab4:
    st.subheader("🔮 Truth Tracker: How good is the AI?")
    st.write("This tab tracks your past predictions and verifies them when the real data becomes available.")
    
    start_verify = st.button("🔄 Verify Accuracy (Check Latest Prices)")
        
    history_df = db_manager.get_all_forecasts()
    
    if not history_df.empty:
        # Normalize column names to match DB schema if needed, but fetch returns DB cols:
        # id, date_logged, ticker, target_date_1d, predicted_1d, ...
        
        # Verify Button Logic
        if start_verify:
            with st.spinner("Checking actual stock prices against predictions..."):
                updates = 0
                # Iterate rows
                for index, row in history_df.iterrows():
                    # Check if actual_1d is null/NaN
                    if pd.isna(row['actual_1d']):
                        target_date = row['target_date_1d']
                        ticker = row['ticker']
                        row_id = row['id']
                        
                        # Find actual price in our main stocks_data
                        stock_records = df[df['Ticker'] == ticker]
                        match = stock_records[stock_records['Date'] == target_date]
                        
                        if not match.empty:
                            actual_price = float(match['Close'].values[0])
                            
                            # Calculate Error %
                            pred = float(row['predicted_1d'])
                            error = abs(actual_price - pred) / actual_price * 100
                            
                            # Update DB
                            db_manager.update_actual_price_1d(row_id, actual_price, error)
                            updates += 1
                
                if updates > 0:
                    st.success(f"Verified {updates} past predictions!")
                    st.rerun() # Refresh to show new data
                else:
                    st.info("No new predictions to verify (Target dates haven't arrived yet or data not fetched).")
        
        # Display Table
        st.dataframe(history_df.sort_values('date_logged', ascending=False))
        
        # Summary Metrics
        if not history_df['error_1d_pct'].isna().all():
            avg_error = history_df['error_1d_pct'].mean()
            st.metric("Average AI Error (1 Day)", f"{avg_error:.2f}%", delta_color="inverse")
            if avg_error < 2.0:
                 st.caption("🔥 The AI is performing excellently (< 2% error).")
            else:
                 st.caption("❄️ The AI needs improvement.")
                 
    else:
        st.info("No history found. Go to 'Forecasting', run a prediction, and click 'Save'.")
