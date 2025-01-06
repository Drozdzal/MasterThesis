from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.config import NUMBER_OF_DAYS, TEST_SIZE, VALIDATION_SIZE, SEED_ID
from predictor.training_model import TrainingModel
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class DataTrainer:
    def __init__(self, model_def: TrainingModel, model_name: str, x: pd.DataFrame, y: pd.DataFrame, number_of_days: int = NUMBER_OF_DAYS):
        self.model = model_def
        self.model_name = model_name
        self.x = x
        self.y = y
        self.n_candles = number_of_days


    def standarize_data(self):
        self.x_standarizer = StandardScaler()
        self.y_standarizer = StandardScaler()
        self.scaled_x = self.x_standarizer.fit_transform(self.x)
        self.y = self.y.values.reshape(-1, 1)
        self.scaled_y = self.y_standarizer.fit_transform(self.y)

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.sequence_X, self.sequence_y, test_size=TEST_SIZE, random_state=SEED_ID, shuffle=False
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_check = self.y_standarizer.inverse_transform(self.y_test)
        self.x_train_nc = self.x_standarizer.inverse_transform(x_train[:,-1,:])
        self.x_test_nc = self.x_standarizer.inverse_transform(x_test[:,-1,:])

    def create_sequences(self):
        self.sequence_X = []
        self.sequence_y = []
        starting_id = self.n_candles
        for i in range(self.n_candles, len(self.x)):
            self.sequence_X.append(self.scaled_x[i - self.n_candles:i])
            self.sequence_y.append(self.scaled_y[i])
        self.sequence_y = np.array(self.sequence_y).astype(np.float32)
        self.sequence_X = np.array(self.sequence_X).astype(np.float32)


    def train_model(self, epochs: int):
        # self.model.fit(self.x_train, self.y_train, validation_split=0.3, epochs=epochs, callbacks=[EarlyStopping(patience=250)])
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=epochs, validation_split = VALIDATION_SIZE, callbacks=[EarlyStopping(patience=50)])

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = self.y_standarizer.inverse_transform(self.y_pred)
        self.y_test = self.y_standarizer.inverse_transform(self.y_test)

    def develop_model(self, epochs: int):
        self.standarize_data()
        self.create_sequences()
        self.split_data()
        self.train_model(epochs)
        self.predict()
        return self.y_pred
