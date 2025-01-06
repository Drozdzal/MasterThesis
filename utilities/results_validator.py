import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import csv

class ResultsValidator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_path = "metrics.csv"

    def calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        correct_direction = np.sum(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1]))
        direction_accuracy = (correct_direction / len(y_true[1:])) * 100

        metrics = {'mae': [round(mae, 3)],
                   'mse': [round(mse, 3)],
                   'r2': [round(r2, 3)],
                   "mape": [round(mape, 3)],
                   "direction_accuracy": [round(direction_accuracy, 3)],
                   "model":[self.model_name]}

        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)
        if not os.path.exists(self.metrics_path):
            # If the file doesn't exist, write the data with the header
            df.to_csv(self.metrics_path, index=False)
        else:
            # If the file exists, append the data without the header
            df.to_csv(self.metrics_path, mode='a', header=False, index=False)
        return metrics

    def plot_results_comparison(self, y_test, predictions, path = None):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price Movement', color='blue')
        plt.plot(predictions, label='Predicted Price Movement', color='red')
        plt.title(f"Price Movement Comparison: {self.model_name}")
        plt.xlabel("Time")
        plt.ylabel("Price Movement")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=30)
        plt.xticks(np.arange(0, len(y_test), step=6), labels=y_test.index[::6])
        plt.gca().tick_params(axis='x', pad=4)  # Decrease the padding value (default is usually around 6)
        plt.ylim(min(y_test["price_movements"]) - 3, max(y_test["price_movements"]) + 3)
        # plt.show()
        if path is None:
            plt.savefig(f"{self.model_name}.png")
        else:
            plt.savefig(f"{path}_short.png")
    def plot_results_with_pre_predictions(self, y_test, predictions, pre_data, file_end = "long", path = None):
        plt.figure(figsize=(12, 6))

        plt.plot(pre_data, label='Pre-Prediction Data', color='green')

        plt.plot(y_test, label='Actual Closing Price',
                 color='blue')

        plt.plot(predictions,
                 label='Predicted Closing Price', color='red')

        plt.title(f"Closing Intel price: {self.model_name}")
        plt.xticks(rotation=30)
        plt.xticks(np.arange(0, len(pre_data)+len(y_test), step=25), labels=pre_data.index[::25].union(y_test.index[::25]))
        plt.gca().tick_params(axis='x', pad=4)  # Decrease the padding value (default is usually around 6)

        plt.xlabel("Time")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.ylim(np.min(np.concatenate((y_test, predictions,pre_data)))- 3, np.max(np.concatenate((y_test, predictions,pre_data)))+3)
        plt.grid()
        # plt.show()
        if path is None:
            plt.savefig(f"{self.model_name}_{file_end}.png")
        else:
            plt.savefig(f"{path}_{file_end}.png")

    def save_results_to_csv(self, mae, mse, r2, mape, direction_accuracy, file_name):
        """
        Save the metrics results to a CSV file in append mode.
        """
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(["Model Name", "MAE", "MSE", "R2", "MAPE", "Direction Accuracy"])

            writer.writerow([self.model_name, mae, mse, r2, mape, direction_accuracy])

    def display_and_save_results(self, y_test, predictions, result_csv_path, pre_data=None):
        """
        Displays the metrics and saves results to CSV.
        Also plots results comparison and pre-prediction data (if available).
        """
        mae, mse, r2, mape, direction_accuracy = self.calculate_metrics(y_test, predictions)

        print(f"Model: {self.model_name}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.4f}%")

        self.plot_results_comparison(y_test, predictions)

        if pre_data is not None:
            self.plot_results_with_pre_predictions(y_test, predictions, pre_data)

        self.save_results_to_csv(mae, mse, r2, mape, direction_accuracy, result_csv_path)



if __name__ == "__main__":
    y_test = np.array([10, 20, 30, 40, 50])  # Actual values
    predictions = np.array([11, 19, 31, 39, 48])  # Predicted values
    pre_data = np.array([5, 6, 7, 8, 9])  # Data before predictions (just an example)

    model_name = "LSTM_Model_1"
    result_csv_path = "lstm_model_results.csv"

    model_results = ResultsValidator(model_name)
    model_results.display_and_save_results(y_test, predictions, result_csv_path, pre_data)
