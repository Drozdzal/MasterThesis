import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
# Load stock data from Yahoo Finance
# Load stock data from Yahoo Finance
ticker = 'AAPL'  # You can replace this with any stock ticker
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Use the 'Close' price for predictions
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Split data into training and test sets (80% training, 20% testing)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare the datasets
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the data for the CNN-LSTM model (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Use the 'Close' price for predictions
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
# Build the CNN-LSTM model
model = Sequential()

# Add a 1D Convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Add an LSTM layer
model.add(LSTM(50, return_sequences=False))

# Add Dropout to avoid overfitting
model.add(Dropout(0.2))

# Add a Dense layer for the final output
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# Make predictions on the test data
predicted_prices = model.predict(X_test)

# Inverse transform to get the actual stock prices
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predicted prices against the actual prices
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction - CNN-LSTM')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Predict future stock price
last_60_days = scaled_data[-time_step:]
last_60_days = last_60_days.reshape(1, time_step, 1)

predicted_price = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_price)

print("Predicted Future Price: ", predicted_price)

