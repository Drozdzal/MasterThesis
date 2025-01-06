from keras import Sequential
from keras.src.layers import Dropout
from keras.src.optimizers import RMSprop

from common.config import NUMBER_OF_SENTIMENT_DATA, NUMBER_OF_DAYS
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adadelta

class LLMLstmModel(TrainingModel):
    @property
    def model_name(self):
        return "LLM_LSTM"

    def model_definition(self,lstm_1: int = 32, lstm_2: int = 16, dense_1: int = 32, number_of_days: int = NUMBER_OF_DAYS):
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(lstm_1, activation='relu', return_sequences=True, input_shape=(NUMBER_OF_DAYS, NUMBER_OF_SENTIMENT_DATA)))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(lstm_2, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))

        # Fully connected layer for prediction
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(1))  # Output for price prediction

        # model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')
        model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8), loss='mae')

        return model
