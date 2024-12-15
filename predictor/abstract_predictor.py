import datetime
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model

class StockPredictor(ABC):
    def generate_model(self):
        self.load_data()
        self.prepare_data()
        self.create_model()
        self.train()
        self.save()
        self.test_model()
        self.visualize_predictions()
    def load_data(self):
        self.df = pd.read_csv("test.csv", index_col=0).sort_index(axis=0, ascending=True)
        data = self.df["Close"].values.reshape(-1, 1)
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        self.train_size = int(len(scaled_data) * 0.8)
        self.train_data = scaled_data[:self.train_size]
        self.X_test = scaled_data[self.train_size:]
        self.Y_test = data[self.train_size:]

    def save(self):
        class_name = self.__class__.__name__

        # Get the current timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine class name and timestamp for the filename
        save_path = f"{class_name}_{current_time}_model.h5"  # For HDF5 format
        self.model.save(save_path)

    def test_model(self):
        self.predictions = self.model.predict(self.X_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)
        self.y_pred = self.df[self.train_size:]["Close"]
        flattened_x = [item for sublist in self.predictions for item in sublist]
        self.y_pred['Predictions'] = pd.Series(np.asarray(flattened_x, float), index=self.y_pred.index)

    def load_model(self):
        self.model = load_model("LSTMPredictor_20241016_222331_model.h5")
    def visualize_predictions(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df[:self.train_size]['Close'], color="yellow",label='Training Data')
        plt.plot(self.y_pred['Predictions'] ,label='Actual Prices')
        # plt.plot(self.predictions, color='red', label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def test_ready(self):
        self.load_data()
        self.prepare_data()
        self.load_model()
        self.test_model()
        self.visualize_predictions()

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def create_model(self):
        pass
    @abstractmethod
    def train(self):
        ...
