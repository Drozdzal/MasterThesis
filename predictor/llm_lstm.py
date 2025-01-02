from common.config import NUMBER_OF_SENTIMENT_DATA, NUMBER_OF_DAYS
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adadelta

class LLMLstmModel(TrainingModel):
    @property
    def model_name(self):
        return "LLM_LSTM"

    @property
    def model_definition(self):
        inputs = Input(shape=(NUMBER_OF_DAYS,NUMBER_OF_SENTIMENT_DATA))
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(128)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(Adadelta(learning_rate=1.0), loss="mse")
        return model
