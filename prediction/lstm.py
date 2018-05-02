"""This file contains the simple one layer LSTM network.

The network is used to predict the next position from a set of previously
observed positions. It can cope with any number of lengths of input sets
but might strugle at larger sizes due to the number of units being
optimised for a set with a length of three.

Example:
    python lstm.py - Generate the json file.
"""

import json
from keras.layers import Input, Dense, LSTM
from keras.models import Model, model_from_json


def conv_lstm(input_shape, units):
    """Recurrent network for the purposes of time series prediction.

    This is a single layer LSTM network with a variable number of units
    and utilises the sigmoid activation function.
    Args:
        input_shape (tripple): Shape of the input tensor for the network.
        units (int): The number of LSTM units in the LSTM layer.
    Returns:
        A Keras model object defining the LSTM network.

    """
    positions = Input(shape=input_shape)
    x = LSTM(units, activation='sigmoid', return_sequences=False)(positions)
    x = Dense(2)(x)
    model = Model(positions, x, name='lstm_prediction')
    return model


if __name__ == "__main__":
    # Set timesteps to be none to allow any number of timesteps
    input_shape = (None, 2)
    units = 55
    model = conv_lstm(input_shape, units)
    model.summary()

    name = "local_positional_one_lstm_one_dense_u-" + str(units) + ".json"

    with open(name, 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    print("Compiled and saved.")
