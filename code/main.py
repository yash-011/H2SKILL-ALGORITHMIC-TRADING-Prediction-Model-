import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from hist_api import get_stock_data
from news_api import get_news_data, filter_irrelevant_news, preprocess_news

def feature_engineering(data, sentiments):
    if data.empty:
        print("Stock data is empty.")
        return data
    
    sentiment_array = []
    for date in data.index.strftime('%Y-%m-%d'):
        sentiment_array.append(sentiments.get(date, 0))  # Use 0 if no sentiment data for the date

    data['Sentiment'] = sentiment_array
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    
    # Debugging: Print the first few rows to inspect intermediate data
    print(data.head(20))
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    if data.empty:
        print("Feature engineering resulted in an empty DataFrame.")
    else:
        print(f"Feature engineering completed. Data shape: {data.shape}")
    
    return data

def build_and_train_model(data):
    features = ['Close', 'Momentum', 'Returns', 'Volatility', 'Sentiment']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    
    seq_length = 10
    X = []
    y = []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1, validation_data=(X_test, y_test))
    
    return model, scaler, X_test, y_test

def make_predictions(model, scaler, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Only inverse transform the 'Close' prices
    y_test_scaled = y_test.reshape(-1, 1)
    predictions_scaled = predictions.reshape(-1, 1)
    
    scaled_true = np.zeros((len(y_test), 5))  # create an array with the same shape used for scaling
    scaled_true[:, 0] = y_test_scaled.flatten()
    y_test = scaler.inverse_transform(scaled_true)[:, 0]
    
    scaled_predictions = np.zeros((len(predictions), 5))  # create an array with the same shape used for scaling
    scaled_predictions[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(scaled_predictions)[:, 0]
    
    return predictions, y_test

def plot_predictions(true_values, predictions):
    plt.figure(figsize=(14,7))
    plt.plot(true_values, color='blue', label='True Values')
    plt.plot(predictions, color='red', label='Predicted Values')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    today = datetime.datetime.today().date()
    thirty_days_ago = today - datetime.timedelta(days=30)
    
    stock_data = get_stock_data('SBIN.NS', start_date=thirty_days_ago.strftime('%Y-%m-%d'), end_date=today.strftime('%Y-%m-%d'))
    
    if stock_data.empty:
        print("No stock data available for the given date range.")
        return
    
    api_key = 'YOUR_MARKETAUX_API_KEY'  # Replace with your Marketaux API key
    news_data = get_news_data(api_key, 'SBI', thirty_days_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    
    keywords = ['SBI', 'State Bank of India', 'SBIN']  # Relevant keywords
    filtered_news = filter_irrelevant_news(news_data, keywords)
    
    sentiments = preprocess_news(filtered_news)
    
    stock_data = feature_engineering(stock_data, sentiments)
    
    if stock_data.empty:
        print("No stock data available after feature engineering.")
        return
    
    model, scaler, X_test, y_test = build_and_train_model(stock_data)
    
    predictions, true_values = make_predictions(model, scaler, X_test, y_test)
    
    comparison_df = pd.DataFrame({
        'True': true_values.flatten(),
        'Predicted': predictions.flatten()
    })
    
    print(comparison_df)
    
    # Plot the predictions
    plot_predictions(true_values, predictions)

if __name__ == "__main__":
    main()
