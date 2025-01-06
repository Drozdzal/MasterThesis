from keras import Sequential
from keras.src.optimizers import RMSprop

from common.config import NUMBER_OF_RAW_DATA, NUMBER_OF_DAYS, NUMBER_OF_SENTIMENT_DATA
from predictor.training_model import TrainingModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adadelta,Adam
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import backend as K


def directional_accuracy_loss(y_true, y_pred):
    """
    Custom loss function for directional accuracy.
    The loss will be 0 if the direction is correct, and a large value if it's wrong.

    Args:
    y_true: The actual values (e.g., actual price movements or returns)
    y_pred: The predicted values (e.g., predicted price movements or returns)

    Returns:
    Loss value: A scalar that is 0 if the direction is correct, otherwise a larger penalty.
    """

    # Calculate the sign of actual price changes (direction of movement)
    true_direction = K.sign(y_true)  # +1 for up, -1 for down

    # Calculate the sign of predicted price changes (direction of movement)
    pred_direction = K.sign(y_pred)  # +1 for up, -1 for down

    # Compare the directions: 1 if they match, 0 if they don't
    correct_direction = K.equal(true_direction, pred_direction)

    # Loss: We penalize incorrect predictions
    loss = K.mean(K.cast(correct_direction, K.floatx()))  # 1 if correct, 0 if wrong

    # Return the loss (subtract from 1 to encourage accuracy)
    return 1 - loss  # We minimize the loss (maximize the directional accuracy)


class LstmModel(TrainingModel):

    @property
    def model_name(self):
        return "LSTM"

    def model_definition(self, lstm_1: int = 16, lstm_2: int = 16, dense_1: int = 16, number_of_days: int = NUMBER_OF_DAYS):
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(lstm_1, activation='relu', return_sequences=True, input_shape=(30,17)))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(lstm_2, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))

        # Fully connected layer for prediction
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(1))  # Output for price prediction

        model.compile(optimizer=RMSprop(learning_rate=0.001), loss="mse")

        return model