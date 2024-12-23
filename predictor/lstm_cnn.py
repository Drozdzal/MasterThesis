from common.config import NUMBER_OF_RAW_DATA
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam


class LstmCnnModel(TrainingModel):
    def __init__(self,
                 time_steps=60,
                 num_features=NUMBER_OF_RAW_DATA,
                 lstm_units=50,
                 cnn_filters=64,
                 kernel_size=3,
                 pool_size=2,
                 dropout_rate=0.2,
                 dense_units=50,
                 learning_rate=0.001,
                 epochs=50,
                 batch_size=32):
        self.time_steps = time_steps
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    @property
    def model_name(self):
        return "LSTM_CNN"

    @property
    def model_definition(self, hp = None):
        inputs = Input(shape=(NUMBER_OF_RAW_DATA,1))

        if hp is None:
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)  # Fixed filters and kernel size
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)  # Fixed filters
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
        else:
            x = Conv1D(filters=hp.Int('cnn_filters', min_value=16, max_value=64, step=16),
                       kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=2),
                       activation='relu')(inputs)
            x = MaxPooling1D(pool_size=hp.Int('pool_size', min_value=2, max_value=4, step=1))(x)
            x = BatchNormalization()(x)

            x = Conv1D(filters=hp.Int('cnn_filters', min_value=16, max_value=64, step=16),
                       kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=2),
                       activation='relu')(x)
            x = MaxPooling1D(pool_size=hp.Int('pool_size', min_value=2, max_value=4, step=1))(x)
            x = BatchNormalization()(x)

        if hp is None:
            x = LSTM(64, return_sequences=False)(x)  # Fixed number of units
        else:
            x = LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=False)(x)

        x = Dropout(0.2)(x)

        if hp is None:
            x = Dense(32, activation='relu')(x)  # Fixed number of units
        else:
            x = Dense(hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu')(x)

        x = Dropout(0.2)(x)

        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        if hp:
            model.compile(
                    optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                    loss='mean_squared_error',
                    metrics=['mae'])
        else:
            model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

        return model
