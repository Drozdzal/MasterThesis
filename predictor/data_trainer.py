from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from predictor.training_model import TrainingModel
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping


class DataTrainer:
    def __init__(self, model: TrainingModel, x: pd.DataFrame, y: pd.DataFrame):
        self.model = model.model_definition
        self.model_name = model.model_name
        self.x = x
        self.y = y

    def standarize_data(self):
        self.x_standarizer = StandardScaler()
        self.y_standarizer = StandardScaler()
        self.scaled_x = self.x_standarizer.fit_transform(self.x)
        self.y = self.y.values.reshape(-1, 1)
        self.scaled_y = self.y_standarizer.fit_transform(self.y)

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.scaled_x, self.scaled_y, test_size=0.2, random_state=42
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def train_model(self, epochs: int):
        self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=epochs, callbacks=[EarlyStopping(patience=100)])

    def predict(self):
        self.y_pred = self.model.predict(self.x_test).flatten()

    def develop_model(self, epochs: int):
        self.standarize_data()
        self.split_data()
        self.train_model(epochs)
        self.predict()
        return self.y_pred
