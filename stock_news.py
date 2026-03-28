import yfinance as yf
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_news_sentiment(ticker_symbol):
    """
    Fetches recent news for the given ticker and calculates sentiment using VADER.
    Returns:
        df: Pandas DataFrame containing the news data, or an empty DataFrame if no news.
        avg_score: Average compound sentiment score (-1 to 1).
        overall_sentiment: String indicating 'Bullish', 'Bearish', or 'Neutral'.
    """
    ticker_symbol = ticker_symbol.upper().strip()
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        news_items = ticker.news
    except Exception as e:
        print(f"Error fetching news for {ticker_symbol}: {e}")
        return pd.DataFrame(), 0.0, "Neutral"

    if not news_items:
        return pd.DataFrame(), 0.0, "Neutral"

    analyzer = SentimentIntensityAnalyzer()
    
    data = []
    total_score = 0.0
    count = 0
    
    for item in news_items:
        # yfinance news API sometimes returns nested dictionaries (content, title, etc)
        content = item.get('content', item) if isinstance(item, dict) else item
        
        title = content.get('title', '')
        if not title:
            continue
            
        provider = content.get('provider') or {}
        publisher = provider.get('displayName', content.get('publisher', 'Unknown'))
        
        click_through = content.get('clickThroughUrl') or {}
        link = click_through.get('url', content.get('link', '#'))
        
        # Parse publish time
        pubDate = content.get('pubDate', '')
        pub_time = content.get('providerPublishTime')
        
        if pubDate:
            # e.g., '2026-03-27T17:47:24Z'
            try:
                date_str = pubDate[:10] + ' ' + pubDate[11:16]
            except:
                date_str = pubDate
        elif pub_time:
            date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
        else:
            date_str = "Unknown"
            
        # Calculate sentiment
        vs = analyzer.polarity_scores(title)
        compound_score = vs['compound']
        
        # Categorize
        if compound_score >= 0.05:
            sentiment_label = "🟢 Bullish"
        elif compound_score <= -0.05:
            sentiment_label = "🔴 Bearish"
        else:
            sentiment_label = "⚪ Neutral"
            
        data.append({
            "Date": date_str,
            "Headline": title,
            "Source": publisher,
            "Sentiment Score": round(compound_score, 3),
            "Sentiment": sentiment_label,
            "Link": link
        })
        
        total_score += compound_score
        count += 1
        
    if count == 0:
         return pd.DataFrame(), 0.0, "Neutral"
         
    df = pd.DataFrame(data)
    
    avg_score = total_score / count
    if avg_score >= 0.05:
        overall = "🟢 Bullish"
    elif avg_score <= -0.05:
        overall = "🔴 Bearish"
    else:
        overall = "⚪ Neutral"
        
    return df, avg_score, overall

if __name__ == "__main__":
    # Test script locally
    df, score, overall = get_news_sentiment("AAPL")
    print(f"Overall Sentiment: {overall} (Score: {score:.3f})")
    print(df.head())
