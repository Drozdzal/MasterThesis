from keras import Sequential

from common.config import NUMBER_OF_RAW_DATA, NUMBER_OF_DAYS, NUMBER_OF_SENTIMENT_DATA
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adadelta,Adam
import keras_tuner as kt

class LstmModel(TrainingModel,kt.HyperModel):

    @property
    def model_name(self):
        return "LSTM"


    @property
    def model_definition(self):
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(NUMBER_OF_DAYS,NUMBER_OF_RAW_DATA)))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(128, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))

        # Fully connected layer for prediction
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))  # Output for price prediction

        model.compile(optimizer=Adadelta(learning_rate=1.0), loss='mae')

        return model

    @property
    def hyper_model(self,hp):
        inputs = Input(
            shape=(hp.Int('time_steps', min_value=50, max_value=100, step=10), 1))  # Adjust time_steps dynamically
        x = LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=256, step=32), return_sequences=True)(inputs)
        x = LSTM(units=hp.Int('lstm_units_2', min_value=16, max_value=128, step=16))(x)
        x = Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16), activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adadelta(learning_rate=1.0), loss='mae')
        return model
