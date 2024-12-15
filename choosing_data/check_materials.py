import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt

# Define assets to analyze: Intel and selected commodities
assets_to_test = ["INTC", "GLD", "SLV", "COPX", "PALL"]  # GLD=Gold, SLV=Silver, COPX=Copper, PALL=Palladium, WOOD=Lumber

# Download data
def download_data():
    data = yf.download(assets_to_test, start="2020-01-01", end="2023-01-01")['Close']
    return data.dropna()

# Make data stationary
def make_stationary(data):
    data_diff = pd.DataFrame()  # Initialize an empty DataFrame for stationary data
    for column in data.columns:
        series = data[column].dropna()  # Drop NaNs from each individual series
        if series.empty:  # Skip if the series is empty
            print(f"Skipping {column} as it contains only NaNs or no data.")
            continue
        p_value = adfuller(series)[1]
        if p_value > 0.05:
            print(f"{column} is not stationary. Applying differencing...")
        series_diff = series.diff().dropna()
        data_diff[column] = series_diff
    return data_diff

# Calculate and visualize correlation matrix
def plot_correlation_matrix(data, title="Correlation Matrix"):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    plt.title("Korelacja INTC oraz surowcow")
    heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)
    plt.show()
    return correlation_matrix

# Perform Granger causality tests to see if each asset affects Intel's future prices
def granger_test_for_assets(data, target='INTC', max_lag=10):
    p_values = {}
    for asset in data.columns:
        if asset == target:
            continue
        test_data = data[[target, asset]].dropna()
        try:
            granger_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            min_p_value = min(granger_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
            p_values[asset] = min_p_value
        except Exception as e:
            print(f"Error testing {asset}: {e}")
            continue
    return p_values

# Main process
data = download_data()
print("\nChecking stationarity of original data...")
data_stationary = make_stationary(data)

# Plot correlation matrix of stationary data
print("\nCorrelation Matrix of Intel and Commodities:")
correlation_matrix = plot_correlation_matrix(data_stationary, title="Intel and Commodity Correlation Matrix")

# Perform Granger causality test and sort results
p_values = granger_test_for_assets(data_stationary)
sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
top_assets = pd.DataFrame(sorted_p_values, columns=['Asset', 'Min P-Value'])

# Display the results
print("\nAssets that may influence Intel's price based on Granger causality:")
print(top_assets)

top_assets.to_csv("rezultat_materialy.csv")