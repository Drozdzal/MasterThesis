from xml.etree.ElementPath import xpath_tokenizer_re

import numpy as np
import pandas as pd
from xgboost import train

from predictor.data_trainer import DataTrainer
from predictor.llm_lstm import LLMLstmModel
from predictor.llm_lstm_cnn import LLMCnnLstmModel
from predictor.lstm import LstmModel
from predictor.lstm_cnn import LstmCnnModel
from utilities.results_validator import ResultsValidator


# Preprocessing and data preparation

# Main execution
if __name__ == "__main__":
    # Load your data here
    data = pd.read_csv("../data/final_3.csv")
    data = data.drop(columns=["Price_Movement"])
    data["Price_Movement"] =  data["INTC_Close"] - data["INTC_Close"].shift(1)
    data["Price_Movement"] = data["Price_Movement"].fillna(0)

    input_columns = list(data.columns)
    input_columns.remove("Price_Movement")
    input_columns.remove("Date")
    x = data[input_columns].drop(columns = 'Unnamed: 0')  # Example data
    y = data["Price_Movement"]  # Example target
    without_sentiment = data[input_columns].drop(
        columns=['Unnamed: 0', "overall_sentiment", "product_sentiment", "quality_of_management",
                 "state_of_competition", "upcoming_events", "semiconductor_sector"])

    num_of_batches = 250
    # lstm_cnn_model = LLMLstmModel()
    # training = DataTrainer(lstm_cnn_model.model_definition,lstm_cnn_model.model_name,without_sentiment,y)
    # y_pred = training.develop_model(num_of_batches)
    # close_prices = training.y_test
    # y_test = training.y_test
    # y_train = training.y_train
    # validator = ResultsValidator(LstmModel().model_name)
    # validator.calculate_metrics(y_test,y_pred)
    # validator.plot_results_comparison(y_test,y_pred)
    # validator.plot_results_with_pre_predictions(y_test,y_pred,y_train)

    # lstm_cnn_model = LstmCnnModel()
    # training = DataTrainer(lstm_cnn_model,without_sentiment,y)
    # y_pred = training.develop_model(num_of_batches)
    # y_test = training.y_test
    # y_train = training.y_train
    # validator = ResultsValidator(lstm_cnn_model.model_name)
    # validator.calculate_metrics(y_test,y_pred)
    # validator.plot_results_comparison(y_test,y_pred)
    # validator.plot_results_with_pre_predictions(y_test,y_pred,y_train)
    # y_test_real = training.x_train_nc["INTC_Close"]
    # y_test_predicted = training.x_train_nc["INTC_Close"]
    # y_test_movements = pd.Series()
    # y_predicted_movements = pd.Series()
    #
    # value = y_test_real.iloc[-1]
    # for movement in y_test:
    #     new_value = value + movement
    #     y_test_movements = y_test_movements._append(pd.Series(new_value))
    #     value = new_value
    #
    # value = y_test_real.iloc[-1]
    # for movement in y_pred:
    #     new_value = value + movement
    #     y_predicted_movements = y_predicted_movements._append(pd.Series(new_value))
    #     value = new_value
    #
    # validator.calculate_metrics(y_test_movements, y_predicted_movements)
    # validator.plot_results_comparison(y_test_movements, y_predicted_movements)
    # validator.plot_results_with_pre_predictions(y_test_movements,y_predicted_movements,y_test_real)


    # lstm_cnn_model = LLMLstmModel()
    # training = DataTrainer(lstm_cnn_model, x, y)
    # y_pred = training.develop_model(num_of_batches)
    # y_test = training.y_test
    # y_train = training.y_train
    # validator = ResultsValidator(lstm_cnn_model.model_name)
    # validator.calculate_metrics(y_test, y_pred)
    # validator.plot_results_comparison(y_test, y_pred)
    # validator.plot_results_with_pre_predictions(y_test, y_pred, y_train)
    # #
    #
    lstm_cnn_model = LLMCnnLstmModel()
    training = DataTrainer(lstm_cnn_model.model_definition, lstm_cnn_model.model_name, x, y)
    y_pred = training.develop_model(num_of_batches)
    y_test = training.y_test
    y_train = training.y_train
    validator = ResultsValidator(lstm_cnn_model.model_name)
    validator.calculate_metrics(y_test, y_pred)
    validator.plot_results_comparison(y_test, y_pred)
    validator.plot_results_with_pre_predictions(y_test, y_pred, y_train)
