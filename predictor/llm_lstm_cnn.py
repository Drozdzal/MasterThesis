from keras.src.optimizers import RMSprop

from common.config import NUMBER_OF_SENTIMENT_DATA, NUMBER_OF_DAYS
from predictor.training_model import TrainingModel

from tensorflow.keras import layers, models


class LLMCnnLstmModel(TrainingModel):
    def __init__(self,
                 time_steps=60,
                 num_features=NUMBER_OF_SENTIMENT_DATA,
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
        return "LLM_LSTM_CNN"

    def model_definition(self, conv_1: int=64, conv_2: int = 128, conv_3: int=64, lstm_1: int=100, lstm_2: int =100, number_of_days: int = NUMBER_OF_DAYS ):
        # Define the model
        model = models.Sequential()

        # Step 1: CNN Layer (Conv1D - over time steps)
        model.add(layers.Conv1D(conv_1, kernel_size=3, activation='relu', padding='same',
                                input_shape=(30, 23)))  # 30 time steps, 23 features
        model.add(layers.MaxPooling1D(pool_size=2))  # MaxPooling to reduce dimensionality

        model.add(layers.Conv1D(conv_2, kernel_size=3, activation='relu',
                                padding='same'))  # Apply convolution on time steps (kernel size 3)
        model.add(layers.MaxPooling1D(pool_size=2))  # MaxPooling to reduce dimensionality

        model.add(layers.Conv1D(conv_3, kernel_size=3, activation='relu', padding='same'))  # Another Conv1D layer
        model.add(layers.MaxPooling1D(pool_size=2))  # MaxPooling to reduce dimensionality

        # Step 2: LSTM Layers
        model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))  # First Bi-LSTM layer
        model.add(layers.Dropout(0.5))  # Dropout for regularization

        model.add(layers.Bidirectional(layers.LSTM(100)))  # Second Bi-LSTM layer
        model.add(layers.Dropout(0.5))  # Dropout for regularization

        # Step 3: Dense Layer (for regression, or classification)
        model.add(layers.Dense(1, activation='linear'))

        # Compile the model with Adam optimizer and MSE loss (for regression tasks)
        model.compile(RMSprop(learning_rate=0.001), loss="mse")

        return model
