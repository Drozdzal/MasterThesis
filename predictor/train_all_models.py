from xml.etree.ElementPath import xpath_tokenizer_re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import train

from common.config import TEST_SIZE, SEED_ID
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
    date=data["Date"]
    date = date.drop(index=date.index[:30])
    date_train, date_test =   train_test_split(
            date, test_size=TEST_SIZE, random_state=SEED_ID, shuffle=False
        )
    input_columns.remove("Date")
    x = data[input_columns].drop(columns = 'Unnamed: 0')  # Example data
    y = data["Price_Movement"]  # Example target
    without_sentiment = data[input_columns].drop(
        columns=['Unnamed: 0', "overall_sentiment", "product_sentiment", "quality_of_management",
                 "state_of_competition", "upcoming_events", "semiconductor_sector"])

    num_of_batches = 250
    for i in range(20,30):
        # lstm_cnn_model = LstmModel()
        # model_name = lstm_cnn_model.model_name+str(i)
        # training = DataTrainer(lstm_cnn_model.model_definition(),model_name,without_sentiment,y)
        # y_pred = training.develop_model(num_of_batches)
        # close_prices = training.y_test
        # y_test = training.y_test
        # y_test = pd.DataFrame(y_test, columns=["price_movements"], index=date_test)
        # y_train = training.y_train
        # y_pred = pd.DataFrame(y_pred, columns=["price_movements"], index=date_test)
        # x_train = training.x_train_nc[:, 0]
        # x_train = pd.DataFrame(x_train, columns=["closed_prices"], index=date_train)
        # x_test = training.x_test_nc[:, 0].reshape(200, 1)
        # x_test = pd.DataFrame(x_test, columns=["closed_prices"], index=date_test)
        # validator = ResultsValidator(model_name)
        # validator.calculate_metrics(y_test, y_pred)
        # validator.plot_results_comparison(y_test, y_pred)
        # validator.plot_results_with_pre_predictions(x_test["closed_prices"] + y_test["price_movements"],
        #                                             x_test["closed_prices"] + y_pred["price_movements"],
        #                                             x_train["closed_prices"])
        #
        # lstm_cnn_model = LstmCnnModel()
        # training = DataTrainer(lstm_cnn_model.model_definition(),model_name,without_sentiment,y)
        # model_name = lstm_cnn_model.model_name+str(i)
        # y_pred = training.develop_model(num_of_batches)
        # y_test = training.y_test
        # y_test = pd.DataFrame(y_test, columns=["price_movements"], index=date_test)
        # y_train = training.y_train
        # y_pred = pd.DataFrame(y_pred, columns=["price_movements"], index=date_test)
        # x_train = training.x_train_nc[:, 0]
        # x_train = pd.DataFrame(x_train, columns=["closed_prices"], index=date_train)
        # x_test = training.x_test_nc[:, 0].reshape(200, 1)
        # x_test = pd.DataFrame(x_test, columns=["closed_prices"], index=date_test)
        # validator = ResultsValidator(model_name)
        # validator.calculate_metrics(y_test, y_pred)
        # validator.plot_results_comparison(y_test, y_pred)
        # validator.plot_results_with_pre_predictions(x_test["closed_prices"] + y_test["price_movements"],
        #                                             x_test["closed_prices"] + y_pred["price_movements"],
        #                                             x_train["closed_prices"])
        #
        # lstm_cnn_model = LLMLstmModel()
        # model_name = lstm_cnn_model.model_name+str(i)
        # training = DataTrainer(lstm_cnn_model.model_definition(),model_name, x, y)
        # y_pred = training.develop_model(num_of_batches)
        # y_test = training.y_test
        # y_test = pd.DataFrame(y_test, columns=["price_movements"], index=date_test)
        # y_train = training.y_train
        # y_pred = pd.DataFrame(y_pred, columns=["price_movements"], index=date_test)
        # x_train = training.x_train_nc[:, 0]
        # x_train = pd.DataFrame(x_train, columns=["closed_prices"], index=date_train)
        # x_test = training.x_test_nc[:, 0].reshape(200, 1)
        # x_test = pd.DataFrame(x_test, columns=["closed_prices"], index=date_test)
        # validator = ResultsValidator(model_name)
        # validator.calculate_metrics(y_test, y_pred)
        # validator.plot_results_comparison(y_test, y_pred)
        # validator.plot_results_with_pre_predictions(x_test["closed_prices"] + y_test["price_movements"],
        #                                             x_test["closed_prices"] + y_pred["price_movements"],
        #                                             x_train["closed_prices"])

        lstm_cnn_model = LLMCnnLstmModel()
        model_name = lstm_cnn_model.model_name+str(i)
        training = DataTrainer(lstm_cnn_model.model_definition(), model_name, x, y)
        y_pred = training.develop_model(num_of_batches)
        y_test = training.y_test
        y_test = pd.DataFrame(y_test,columns=["price_movements"],index = date_test)
        y_train = training.y_train
        y_pred = pd.DataFrame(y_pred,columns=["price_movements"],index = date_test)
        x_train = training.x_train_nc[:,0]
        x_train = pd.DataFrame(x_train, columns=["closed_prices"], index =date_train)
        x_test = training.x_test_nc[:,0].reshape(200,1)
        x_test =  pd.DataFrame(x_test,columns=["closed_prices"],index = date_test)
        validator = ResultsValidator(model_name)
        validator.calculate_metrics(y_test, y_pred)
        validator.plot_results_comparison(y_test, y_pred)
        validator.plot_results_with_pre_predictions(x_test["closed_prices"]+y_test["price_movements"], x_test["closed_prices"]+y_pred["price_movements"], x_train["closed_prices"])
