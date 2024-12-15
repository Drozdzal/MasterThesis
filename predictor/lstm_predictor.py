import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from predictor.abstract_predictor import StockPredictor


class LSTMPredictor(StockPredictor):
    def prepare_data(self):
        X, Y = [], []
        time_step = 60
        for i in range(len(self.train_data) - time_step - 1):
            a = self.train_data[i:(i + time_step), 0]
            X.append(a)
            Y.append(self.train_data[i + time_step, 0])
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))


    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self):
        self.model.fit(self.X_train, self.Y_train, batch_size=64 ,epochs=50)

predictor = LSTMPredictor()
predictor.generate_model()
# predictor.test_ready()