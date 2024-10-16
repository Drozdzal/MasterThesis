import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download stock data (e.g., Apple)
# Download stock data (e.g., Apple)
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Use the 'Close' price for the prediction task
data = data[['Close']]

# Convert to numpy array
data_values = data.values

# Scaling the data to be in range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# Create features and labels from the data
def create_features(data, time_step):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

# Set the time step to 60 (using 60 previous days to predict the next day)
time_step = 60
X, Y = create_features(scaled_data, time_step)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape data to 2D for XGBoost (samples, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Initialize the XGBoost Regressor model
xgboost_model = XGBRegressor(n_estimators=1000, learning_rate=0.01)

# Train the model
xgboost_model.fit(X_train, y_train)

# Make predictions
y_pred = xgboost_model.predict(X_test)


# Inverse scale the predicted and actual values to original scale
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error: {mse}")

# Plot the predicted prices vs actual prices
plt.figure(figsize=(10,5))
plt.plot(y_test_scaled, color='blue', label='Actual Stock Prices')
plt.plot(y_pred_scaled, color='red', label='Predicted Stock Prices')
plt.title('Stock Price Prediction using XGBoost')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# Predict the next day's stock price
last_60_days = scaled_data[-60:]  # Take the last 60 days of data
last_60_days = last_60_days.reshape(1, -1)  # Reshape to be 1 sample, 60 features

predicted_next_day = xgboost_model.predict(last_60_days)
predicted_next_day_scaled = scaler.inverse_transform(predicted_next_day.reshape(-1, 1))

print("Predicted Next Day Stock Price:", predicted_next_day_scaled)
