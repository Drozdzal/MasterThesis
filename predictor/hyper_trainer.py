import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.config import NUMBER_OF_DAYS
from predictor.lstm import hyper_model
from predictor.training_model import TrainingModel
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import tensorflow as tf
from keras.src.callbacks import EarlyStopping


class HyperTrainer:
    def __init__(self, model: TrainingModel, x: pd.DataFrame, y: pd.DataFrame):
        self.model_name = model.model_name
        self.x = x
        self.y = y
        self.n_candles = NUMBER_OF_DAYS
        self.tuner = kt.RandomSearch(
            hyper_model,
            objective='val_loss',  # Minimize the validation loss
            max_trials=10,  # Number of different hyperparameter combinations to try
            executions_per_trial=2,  # Number of models to train for each set of hyperparameters
            directory='lstm_2_lstm',  # Directory to store the results
            project_name='lstm_tuning'
        )

    def standarize_data(self):
        self.x_standarizer = StandardScaler()
        self.y_standarizer = StandardScaler()
        self.scaled_x = self.x_standarizer.fit_transform(self.x)
        self.y = self.y.values.reshape(-1, 1)
        self.scaled_y = self.y_standarizer.fit_transform(self.y)

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.sequence_X, self.sequence_y, test_size=0.1, random_state=42
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def create_sequences(self):
        self.sequence_X = []
        self.sequence_y = []

        for i in range(self.n_candles, len(self.x)):
            self.sequence_X.append(self.scaled_x[i - self.n_candles:i])
            self.sequence_y.append(self.scaled_y[i])
        self.sequence_y = np.array(self.sequence_y).astype(np.float32)
        self.sequence_X = np.array(self.sequence_X).astype(np.float32)


    def train_model(self, epochs: int):
        # self.model.fit(self.x_train, self.y_train, validation_split=0.3, epochs=epochs, callbacks=[EarlyStopping(patience=250)])
        stop_early = EarlyStopping(monitor='val_loss', patience=10)
        self.tuner.search(self.x_train, self.y_train, epochs=20, batch_size=32, validation_split=0.2,
                     callbacks=[stop_early])

        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)

    def develop_models(self, epochs: int):
        self.standarize_data()
        self.create_sequences()
        self.split_data()
        self.train_model(epochs)

