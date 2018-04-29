"""LSTM model for next frame prediction."""

import json
from keras.layers import Input, Dense, LSTM
from keras.models import Model

import keras.models as models


def conv_lstm(input_shape_1, units):
    """Generate a convolutional LSTM network, TF format."""
    # Local Positional Prediction
    positions = Input(shape=input_shape_1)
    x = LSTM(units, activation='sigmoid', return_sequences=False)(positions)
    x = Dense(2)(x)
    # Output
    model = Model(positions, x, name='lstm_prediction')
    return model


if __name__ == "__main__":
    # Set timesteps to be none
    input_shape = (None, 2)
    units = 55
    model = conv_lstm(input_shape, units)
    model.summary()
    name = "local_positional_one_lstm_one_dense_u-" + str(units) + ".json"
    with open(name, 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    with open(name, 'r') as outfile:
        model_test = models.model_from_json(outfile.read())
    print("Compiled and saved.")
