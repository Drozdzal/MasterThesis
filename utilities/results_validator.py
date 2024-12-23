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
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        correct_direction = np.sum(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1]))
        direction_accuracy = (correct_direction / len(y_true[1:])) * 100

        # Example metrics
        metrics = {'mae': mae,
                   'mse': mse,
                   'r2': r2,
                   "mape": mape,
                   "direction_accuracy": direction_accuracy}

        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)

        df.to_csv('metrics.csv', mode='a', header=False, index=False)

    def plot_results_comparison(self, y_test, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price Movement', color='blue')
        plt.plot(predictions, label='Predicted Price Movement', color='red')
        plt.title(f"Price Movement Comparison: {self.model_name}")
        plt.xlabel("Time")
        plt.ylabel("Price Movement")
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig(f"{self.model_name}_short.png")
    def plot_results_with_pre_predictions(self, y_test, predictions, pre_data):
        plt.figure(figsize=(12, 6))

        plt.plot(pre_data, label='Pre-Prediction Data', color='green')

        plt.plot(np.arange(len(pre_data), len(pre_data) + len(y_test)), y_test, label='Actual Price Movement',
                 color='blue')

        plt.plot(np.arange(len(pre_data), len(pre_data) + len(predictions)), predictions,
                 label='Predicted Price Movement', color='red')

        plt.title(f"Price Movement Comparison (With Pre-Prediction Data): {self.model_name}")
        plt.xlabel("Time")
        plt.ylabel("Price Movement")
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig(f"{self.model_name}_long.png")

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
