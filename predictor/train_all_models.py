from xml.etree.ElementPath import xpath_tokenizer_re

import numpy as np
import pandas as pd
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
    input_columns = list(data.columns)
    input_columns.remove("Price_Movement")
    input_columns.remove("Date")
    x = data[input_columns].drop(columns = 'Unnamed: 0')  # Example data
    y = data["Price_Movement"]  # Example target
    without_sentiment = data[input_columns].drop(
        columns=['Unnamed: 0', "overall_sentiment", "product_sentiment", "quality_of_management",
                 "state_of_competition", "upcoming_events", "semiconductor_sector"])


    num_of_batches = 100
    lstm_cnn_model = LstmModel()
    training = DataTrainer(lstm_cnn_model,without_sentiment,y)
    y_pred = training.develop_model(num_of_batches)
    y_test = training.y_test
    y_train = training.y_train
    validator = ResultsValidator(LstmModel().model_name)
    validator.calculate_metrics(y_test,y_pred)
    validator.plot_results_comparison(y_test,y_pred)
    validator.plot_results_with_pre_predictions(y_test,y_pred,y_train)

    lstm_cnn_model = LstmCnnModel()
    training = DataTrainer(lstm_cnn_model,without_sentiment,y)
    y_pred = training.develop_model(num_of_batches)
    y_test = training.y_test
    y_train = training.y_train
    validator = ResultsValidator(lstm_cnn_model.model_name)
    validator.calculate_metrics(y_test,y_pred)
    validator.plot_results_comparison(y_test,y_pred)
    validator.plot_results_with_pre_predictions(y_test,y_pred,y_train)



    lstm_cnn_model = LLMLstmModel()
    training = DataTrainer(lstm_cnn_model, x, y)
    y_pred = training.develop_model(num_of_batches)
    y_test = training.y_test
    y_train = training.y_train
    validator = ResultsValidator(lstm_cnn_model.model_name)
    validator.calculate_metrics(y_test, y_pred)
    validator.plot_results_comparison(y_test, y_pred)
    validator.plot_results_with_pre_predictions(y_test, y_pred, y_train)


    lstm_cnn_model = LLMCnnLstmModel()
    training = DataTrainer(lstm_cnn_model, x, y)
    y_pred = training.develop_model(num_of_batches)
    y_test = training.y_test
    y_train = training.y_train
    validator = ResultsValidator(lstm_cnn_model.model_name)
    validator.calculate_metrics(y_test, y_pred)
    validator.plot_results_comparison(y_test, y_pred)
    validator.plot_results_with_pre_predictions(y_test, y_pred, y_train)
