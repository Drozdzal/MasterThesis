import itertools
import os

import pandas as pd

from predictor.data_trainer import DataTrainer
from predictor.llm_lstm import LLMLstmModel
from predictor.llm_lstm_cnn import LLMCnnLstmModel
from predictor.lstm import LstmModel
from predictor.lstm_cnn import LstmCnnModel
from predictor.training_model import TrainingModel
from itertools import product

from utilities.results_validator import ResultsValidator


class ModelTuner:
    def __init__(self, model: TrainingModel, parameters: dict):
        self.model = model
        self.parameters = parameters
        self.file_name = model.model_name
        self._create_combinations()
        data = pd.read_csv("../data/final_3.csv")
        data = data.drop(columns=["Price_Movement"])
        data["Price_Movement"] = data["INTC_Close"] - data["INTC_Close"].shift(1)
        data["Price_Movement"] = data["Price_Movement"].fillna(0)

        input_columns = list(data.columns)
        input_columns.remove("Price_Movement")
        input_columns.remove("Date")
        self.raw_x = data[input_columns].drop(columns='Unnamed: 0')  # Example data
        self.raw_y = data["Price_Movement"]
        self.without_sentiment = data[input_columns].drop(
        columns=['Unnamed: 0', "overall_sentiment", "product_sentiment", "quality_of_management",
                 "state_of_competition", "upcoming_events", "semiconductor_sector"])
        self.path = f"{self.model.model_name}/tuning.csv"

    def _create_combinations(self):
        param_combinations = itertools.product(*self.parameters.values())

        # Convert each combination into a dictionary
        self.combinations = [
            dict(zip(self.parameters.keys(), combination))
            for combination in param_combinations
        ]

    def search(self):
        i=0
        epochs = 250
        for combination in self.combinations:
            combination_number = i
            model_def = self.model.model_definition(**combination)
            training = DataTrainer(model_def,self.model.model_name, self.raw_x, self.raw_y, number_of_days=combination["number_of_days"])
            y_pred = training.develop_model(epochs)
            close_prices = training.y_test
            y_test = training.y_test
            y_train = training.y_train
            name = f"{model.model_name}_{combination_number}"
            validator = ResultsValidator(name)
            metrics = validator.calculate_metrics(y_test, y_pred)
            image_path = f"{self.model.model_name}/{combination_number}"
            validator.plot_results_comparison(y_test, y_pred, path = image_path)
            validator.plot_results_with_pre_predictions(y_test, y_pred, y_train, path = image_path)
            i+=1
            for key,value in combination.items():
                metrics[key] = value
            df = pd.DataFrame(metrics)
            if not os.path.exists(self.path):
                # If the file doesn't exist, write the data with the header
                df.to_csv(self.path, index=False)
            else:
                # If the file exists, append the data without the header
                df.to_csv(self.path, mode='a', header=False, index=False)
            del training
            del validator


# params = {"lstm_1": [16, 32, 64, 128],
#  "lstm_2": [16, 32, 64, 128],
#  "dense_1": [16, 32, 64],
#  "number_of_days": [10, 20, 30, 50]}


params = {"lstm_1": [32, 64, 128],
 "lstm_2": [32, 64, 128],
 "conv_1": [32,64,128],
 "conv_2": [32,64,128],
 "conv_3": [32,64,128],
 "number_of_days": [20, 30, 50]}

model = LLMCnnLstmModel()
tuner = ModelTuner(model,params)
tuner.search()