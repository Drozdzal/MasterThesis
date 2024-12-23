from common.config import NUMBER_OF_RAW_DATA
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

class LstmModel(TrainingModel):

    @property
    def model_name(self):
        return "LSTM"

    @property
    def model_definition(self):
        inputs = Input(shape=(NUMBER_OF_RAW_DATA,1))
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        return model

    @property
    def hyper_model(self,hp):
        inputs = Input(
            shape=(hp.Int('time_steps', min_value=50, max_value=100, step=10), 1))  # Adjust time_steps dynamically
        x = LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
        x = LSTM(units=hp.Int('lstm_units_2', min_value=16, max_value=64, step=16))(x)
        x = Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
            loss='mse')
        return model
