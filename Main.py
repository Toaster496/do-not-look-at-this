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
import ccxt
import time
import threading
# Initialize Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Global variables
scaler = MinMaxScaler(feature_range=(0, 1))
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 53)),  # Adjusted for 53 features (price, sentiment, HMM states, and indicators)
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

# List of NOT 50 indicators
INDICATORS = [
    "sma", "ema", "macd", "adx", "stochastic",
    "bollinger_bands_upper", "bollinger_bands_lower",
]


def calculate_indicators(df):
    # Moving Averages
    df['sma'] = df['close'].rolling(window=14).mean()
    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()
    df['wma'] = talib.WMA(df['close'], timeperiod=14)  # Weighted Moving Average

    df['macd'] = macd - macdsignal
    df['stochastic'] = 100 * ((df['close'] - df['low'].rolling(window=14).min()) /
                              (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()))
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

    # Bollinger Bands
    upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bollinger_bands_upper'] = upper_band
    df['bollinger_bands_lower'] = lower_band

    # Remove or skip unsupported/complex indicators
    skipped_indicators = [
        "supertrend", "pivot_points", "fibonacci_retracement", "donchian_channels", "keltner_channels",
        "volume_profile", "trend_strength_index", "gator_oscillator", "mass_index",
        "schaff_trend_cycle", "ease_of_movement", "chaikin_volatility"
    ]
    for indicator in skipped_indicators:
        if indicator in df.columns:
            del df[indicator]

    return dfM

# Swing Breakout Sequence (SBS) Trainer and Identifier
def identify_sbs(df):
    df['high_low_diff'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['sbs_signal'] = ((df['high_low_diff'] > df['close_open_diff']) & (df['close_open_diff'] > 0)).astype(int)
    return df

# Data Processing
def fetch_market_data():
    global data
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=500)
    df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    df = calculate_indicators(df)
    df = identify_sbs(df)
    data = df

# Sentiment Analysis
def fetch_news_sentiment():
    url = "https://newsapi.org/v2/everything?q=bitcoin&apiKey=YOUR_API_KEY"
    response = requests.get(url).json()
    sentiment_scores = []
    for article in response['articles']:
        sentiment_scores.append(sia.polarity_scores(article['title'])['compound'])
    return np.mean(sentiment_scores)

# HMM Model with SBS (Swing Breakout Sequence)
def calculate_hmm_states():
    global data
    if data is not None:
        close_prices = data['close'].values.reshape(-1, 1)
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        hmm.fit(close_prices)
        states = hmm.predict(close_prices)
        return states
    return None

# Combined Feature Preparation
def prepare_training_data():
    global data
    if data is not None:
        sentiment = fetch_news_sentiment()
        hmm_states = calculate_hmm_states()

        if hmm_states is not None:
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
            return np.array(features), close_data[60:]
    return None, None

# Train LSTM Model
def train_model():
    global data
    features, target = prepare_training_data()
    if features is not None and target is not None:
        train_x = features
        train_y = target
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2]))
        model.fit(train_x, train_y, batch_size=1, epochs=1)

# Trading Bot Functionality
def trading_bot():
    global trading, predicted_price
    fetch_market_data()
    train_model()

    while trading:
        latest_data = data['close'].values[-60:]
        sentiment = fetch_news_sentiment()
        hmm_states = calculate_hmm_states()[-60:]
        indicators_data = data[INDICATORS].values[-60:]

        if hmm_states is not None:
            latest_features = np.column_stack((
                latest_data,
                [sentiment] * 60,
                hmm_states,
                indicators_data
            ))
            latest_features = np.reshape(latest_features, (1, latest_features.shape[0], latest_features.shape[1]))
            predicted_price = model.predict(latest_features)[0, 0]

        time.sleep(60)

# Streamlit App
def main():
    global trading

    st.title("Advanced AI-Driven Trading Bot with Sentiment, HMM, SBS, and Indicators")

    if st.button("Start Trading"):
        if not trading:
            trading = True
            threading.Thread(target=trading_bot, daemon=True).start()
            st.success("Trading started!")

    if st.button("Stop Trading"):
        trading = False
        st.warning("Trading stopped.")

    st.header("Predicted Price")
    st.write(predicted_price)

if __name__ == "__main__":
    main()
