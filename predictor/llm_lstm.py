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

    def model_definition(self,lstm_1: int = 64, lstm_2: int = 64, dense_1: int = 32, number_of_days: int = NUMBER_OF_DAYS):
        inputs = Input(shape=(number_of_days,NUMBER_OF_SENTIMENT_DATA))
        x = LSTM(lstm_1, return_sequences=True)(inputs)
        x = LSTM(lstm_2)(x)
        x = Dense(dense_1, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(RMSprop(learning_rate=0.001), loss="mse")
        return model
