from common.config import NUMBER_OF_SENTIMENT_DATA
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

class LLMLstmModel(TrainingModel):
    @property
    def model_name(self):
        return "LLM_LSTM"

    @property
    def model_definition(self):
        inputs = Input(shape=(NUMBER_OF_SENTIMENT_DATA,1))
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        return model
