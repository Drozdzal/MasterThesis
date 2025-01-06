import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import seaborn as sns

stocks_to_test = ["AMD", "NVDA", "TSM", "MSFT", "AAPL", "GOOGL", "IBM", "TXN", "QCOM",
                  "AMZN", "MU", "ADBE", "META", "ORCL", "CSCO", "SPY"]

def download_data():
    tickers = ["INTC", *stocks_to_test]
    data = yf.download(tickers, start="2020-10-01", end="2024-10-01")['Close']
    return data.dropna(axis=1, how='all')  # Drop columns with all NaN values

def make_stationary(data):
    data_diff = pd.DataFrame()
    for column in data.columns:
        series = data[column].dropna()
        if len(series) == 0:
            print(f"Skipping {column} due to insufficient data.")
            continue

        p_value = adfuller(series)[1]
        if p_value > 0.05:
            print(f"{column} is not stationary. Applying differencing...")
        series_diff = series.diff().dropna()
        data_diff[column] = series_diff
    return data_diff

# Perform Granger causality tests to see if each stock affects Intel's future prices
def granger_test_for_stocks(data, target='INTC', max_lag=10):
    p_values = {}
    for stock in stocks_to_test:
        if stock not in data.columns:
            print(f"Skipping {stock} as it is missing in the data.")
            continue
        test_data = data[[target, stock]].dropna()
        if test_data.empty:
            print(f"Skipping {stock} due to insufficient data.")
            continue

        try:
            granger_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            min_p_value = min(granger_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
            p_values[stock] = min_p_value
        except Exception as e:
            print(f"Error testing {stock}: {e}")
            continue
    return p_values

def plot_correlation_matrix(data, title="Correlation Matrix"):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    plt.title(title)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, cbar=True)

    plt.show()
# Main process
data = download_data()
print("\nChecking stationarity of original data...")

# data_stationary = make_stationary(data)
# data_stationary.to_csv("dane_20.csv")
# Perform Granger causality test and sort results

# data_stationary = pd.read_csv("dane_20.csv")
# p_values = granger_test_for_stocks(data_stationary)
# sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
# top_20_stocks = pd.DataFrame(sorted_p_values[:20], columns=['Stock', 'Min P-Value'])

# top_20_stocks.to_csv("rezultat_grangera.csv")
# # Display the results
# print("\nTop 20 stocks that may influence Intel's price the most based on Granger causality:")
# print(top_20_stocks)

plot_correlation_matrix(data)
