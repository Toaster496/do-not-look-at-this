import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
import time
import threading

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize Sentiment Analyzer
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Global variables
scaler = MinMaxScaler(feature_range=(0, 1))
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 10)),  # 60 time steps, 10 features
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

data = None
predicted_price = 0
trading = False

# Indicators including volatility
INDICATORS = ["sma", "ema", "macd", "stochastic", 
              "bollinger_upper", "bollinger_lower", "volatility"]

def calculate_indicators(df):
    # Price transformations
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=14).std()  # New feature: volatility
    
    # Moving Averages
    df['sma'] = df['close'].rolling(window=14).mean()
    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # MACD (using EMAs)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    
    # Stochastic Oscillator
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['stochastic'] = 100 * ((df['close'] - low14) / (high14 - low14))
    
    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = sma20 + (2 * std20)
    df['bollinger_lower'] = sma20 - (2 * std20)
    
    # Drop NaN values from indicators
    return df.dropna()

def identify_sbs(df):
    df['high_low_diff'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['sbs_signal'] = ((df['high_low_diff'] > df['close_open_diff']) & 
                       (df['close_open_diff'] > 0)).astype(int)
    return df

def fetch_market_data():
    global data
    try:
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period="7d", interval="1m")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df['close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))  # Scale close prices
        df = calculate_indicators(df)
        df = identify_sbs(df)
        data = df
    except Exception as e:
        print(f"Data fetch error: {e}")

def fetch_news_sentiment():
    url = "https://newsapi.org/v2/everything?q=bitcoin&apiKey=9a947fd72c6049be99a160355c70adbc"
    try:
        response = requests.get(url).json()
        sentiment_scores = [
            sia.polarity_scores(article['title'])['compound']
            for article in response.get('articles', [])
        ]
        return np.mean(sentiment_scores) if sentiment_scores else 0
    except Exception as e:
        print(f"Sentiment error: {e}")
        return 0

def calculate_hmm_states():
    global data
    if data is not None and not data.empty:
        close_prices = data['close'].values.reshape(-1, 1)
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        hmm.fit(close_prices)
        return hmm.predict(close_prices)
    return np.zeros(len(data)) if data is not None else None

def prepare_training_data():
    global data
    if data is not None and len(data) > 60:
        sentiment = fetch_news_sentiment()
        hmm_states = calculate_hmm_states()
        
        features = []
        close_data = data['close'].values
        indicators_data = data[INDICATORS].values
        
        for i in range(60, len(close_data)):
            feature_vector = np.column_stack((
                close_data[i-60:i],
                [sentiment] * 60,
                hmm_states[i-60:i],
                indicators_data[i-60:i]
            ))
            features.append(feature_vector)
        
        features_array = np.array(features)
        print(f"Final features shape: {features_array.shape}")  # Debugging line
        return features_array, close_data[60:]
    return None, None

def train_model():
    features, target = prepare_training_data()
    if features is not None and target is not None:
        print("Training model...")
        history = model.fit(features, target, batch_size=1, epochs=5, verbose=1)  # Train for 5 epochs
        print(f"Training loss: {history.history['loss'][-1]:.4f}")

def trading_bot():
    global trading, predicted_price
    while trading:
        try:
            fetch_market_data()
            train_model()
            
            if data is not None and len(data) >= 60:
                latest_data = scaler.transform(data['close'].values[-60:].reshape(-1, 1))  # Scale latest data
                sentiment = fetch_news_sentiment()
                hmm_states = calculate_hmm_states()[-60:]
                indicators_data = data[INDICATORS].values[-60:]
                
                latest_features = np.column_stack((
                    latest_data,
                    [sentiment] * 60,
                    hmm_states,
                    indicators_data
                )).reshape(1, 60, -1)
                
                print(f"Latest features shape: {latest_features.shape}")  # Debugging line
                predicted_price = model.predict(latest_features, verbose=0)[0, 0]
                print(f"Predicted price: {predicted_price}")  # Debugging line
                
        except Exception as e:
            print(f"Trading error: {e}")
        time.sleep(60)

def main():
    global trading
    st.title("AI Trading Bot with Yahoo Finance")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Trading") and not trading:
            trading = True
            threading.Thread(target=trading_bot, daemon=True).start()
            st.success("Trading started!")
    
    with col2:
        if st.button("Stop Trading"):
            trading = False
            st.warning("Trading stopped.")
    
    st.subheader("Real-time Prediction")
    st.metric("Predicted Price", f"{predicted_price:.4f}")

if __name__ == "__main__":
    main()
