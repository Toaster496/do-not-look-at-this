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
import os
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define the folder patht
MODEL_DIR = "/content/drive/My Drive/cryptoBot"

# Ensure the directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model save path
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.h5")
best_loss = float('inf')  # Start with a high loss

def train_model():
    global best_loss

    features, target = prepare_training_data()
    if features is not None and target is not None:
        print("Training model...")
        history = model.fit(features, target, batch_size=1, epochs=5, verbose=1)

        current_loss = history.history['loss'][-1]  # Get latest training loss
        print(f"Training loss: {current_loss:.4f}")

        # Save model only if it has a lower loss
        if current_loss < best_loss:
            best_loss = current_loss
            model.save(MODEL_SAVE_PATH)
            print(f"New best model saved at {MODEL_SAVE_PATH} with loss: {best_loss:.4f}")

# Suppress TensorFlow logging
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

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1339511103482236959/r0cJaoMtY9mZiOCeLu6Rj6NUN9rHMydW0QtDG_N4qDolkNiyYpUkh4aCPONH9rm2YbmY"

def calculate_probability():
    """Calculates probability based on MACD, volatility, sentiment, and HMM states."""
    global data

    # 1️⃣ **MACD Strength**
    macd_strength = abs(data['macd'].iloc[-1])  
    macd_prob = min(100, macd_strength * 10)  # Normalize to 0-100%

    # 2️⃣ **Volatility Confidence** (Lower volatility = Higher confidence)
    volatility = data['volatility'].iloc[-1]
    volatility_prob = max(0, 100 - (volatility * 200))  # Normalize inversely

    # 3️⃣ **Sentiment Score**  
    sentiment = fetch_news_sentiment()
    sentiment_prob = ((sentiment + 1) / 2) * 100  # Convert -1 to 1 range into 0-100%

    # 4️⃣ **HMM Market State** (Higher confidence if trending)
    hmm_states = calculate_hmm_states()
    recent_state = hmm_states[-1] if len(hmm_states) > 0 else 0
    hmm_prob = (recent_state / 2) * 100  # Convert 0-2 HMM states to 0-100%

    # 🎯 **Final Probability Calculation** (Equal Weighting)
    final_probability = np.mean([macd_prob, volatility_prob, sentiment_prob, hmm_prob])
    return round(final_probability, 2)

def send_discord_alert(predicted_price, last_price, probability):
    """Sends AI prediction updates to Discord via Webhook."""
    price_change = ((predicted_price - last_price) / last_price) * 100  # % Change
    direction = "📈 **UP**" if price_change > 0 else "📉 **DOWN**"

    message = {
        "username": "AI Trading Bot",
        "avatar_url": "https://i.imgur.com/YX4xN6Z.png",  # Custom bot avatar
        "embeds": [
            {
                "title": "Bitcoin Price Prediction",
                "description": f"AI Model Prediction: {direction}\n\n"
                               f"💰 **Predicted Price**: ${predicted_price:.2f}\n"
                               f"📊 **Price Change**: {price_change:.2f}%\n"
                               f"🎯 **Probability**: {probability:.2f}%",
                "color": 3066993 if price_change > 0 else 15158332  # Green for UP, Red for DOWN
            }
        ]
    }
    
    response = requests.post(DISCORD_WEBHOOK_URL, json=message)
    if response.status_code == 204:
        print("✅ Discord alert sent!")
    else:
        print(f"❌ Failed to send Discord alert: {response.status_code}, {response.text}")

def trading_bot():
    global trading, predicted_price
    last_price = None

    while trading:
        try:
            fetch_market_data()
            train_model()
            
            if data is not None and len(data) >= 60:
                latest_close = data['close'].values[-60:].reshape(-1, 1)
                latest_scaled = scaler.transform(latest_close)

                sentiment = fetch_news_sentiment()
                hmm_states = calculate_hmm_states()[-60:]
                indicators_data = data[INDICATORS].values[-60:]

                latest_features = np.column_stack((
                    latest_scaled, [sentiment] * 60, hmm_states, indicators_data
                )).reshape(1, 60, -1)

                raw_prediction = model.predict(latest_features, verbose=0)[0, 0]
                predicted_price = scaler.inverse_transform([[raw_prediction]])[0, 0]

                # **Compute final probability using all confidence factors**
                probability = calculate_probability()

                print(f"Predicted BTC Price: ${predicted_price:.2f} | Probability: {probability:.2f}%")

                # **Send alert to Discord**
                if last_price:
                    send_discord_alert(predicted_price, last_price, probability)

                last_price = predicted_price  # Update last price

        except Exception as e:
            print(f"Trading error: {e}")

        time.sleep(60)  # Run every minute


if __name__ == "__main__":
    trading = True
    print("Starting Trading Bot...")

    # Start Trading Thread
    threading.Thread(target=trading_bot, daemon=True).start()

    # Keep the script running
    while trading:
        time.sleep(60)
