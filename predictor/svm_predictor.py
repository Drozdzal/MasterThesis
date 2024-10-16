import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Load stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Use the 'Close' price for predictions
data = data[['Close']]

# Convert the data to numpy array
data_values = data.values

# Scale the data between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# Create features and labels
def create_features(data, time_step):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

# Use 60 previous days to predict the next day (time_step = 60)
time_step = 60
X, Y = create_features(scaled_data, time_step)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape the data for SVR (2D input)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Initialize the SVR model with radial basis function (RBF) kernel
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svr_model.predict(X_test)
# Inverse scale the predictions and actual values to get them back to original scale
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error: {mse}")

# Plot the predicted vs actual stock prices
plt.figure(figsize=(10,5))
plt.plot(y_test_scaled, color='blue', label='Actual Stock Prices')
plt.plot(y_pred_scaled, color='red', label='Predicted Stock Prices')
plt.title('Stock Price Prediction using SVM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# Predict future stock price
last_60_days = scaled_data[-60:]  # Last 60 days of data
last_60_days = last_60_days.reshape(1, -1)  # Reshape to be 1 sample, 60 features

predicted_next_day = svr_model.predict(last_60_days)
predicted_next_day_scaled = scaler.inverse_transform(predicted_next_day.reshape(-1, 1))

print("Predicted Next Day Stock Price:", predicted_next_day_scaled)
