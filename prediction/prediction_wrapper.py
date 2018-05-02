"""Wrapper to abstract away the LSTM network.

This file contains the PredictionWrapper class which encapsulates the
lstm network and stops any client from having to deal with loading the network
or knowing what weights to use.

"""

import keras.models as models
from keras.optimizers import Adam
import numpy as np
# import pickle


MODEL_PATH = "./model/local_positional_one_lstm_one_dense_u-128.json"
WEIGHT_PATH = "./model/prediction_weights_2018-04-30_13_24.hdf5"


class PredictionWrapper:
    """Simple wrapper class for the LSTM network.

    This class contains the PredictionWrapper class which encapsulates the
    lstm network and stops any client program from having to deal with loading
    the network or knowing what weights to use.
    Attributes:
        lstm (Keras.Model): Instance of the LSTM prediction model

    """

    def __init__(self):
        """Create a new wrapper instance.

        The constructor the for PredictionWrapper class. Appart from
        initalising a new intance of the wrapper class, this constuctor also
        creates a new lstm prediciton instance which is then allocated on the
        GPU.
        Args:
            None

        """
        # Create network.
        loss = "mean_squared_error"
        optmzr = Adam(lr=0.0001)

        # Load the model from JSON.
        with open(MODEL_PATH) as model_file:
            self.lstm = models.model_from_json(model_file.read())

        # Compile and load weights.
        self.lstm .compile(loss=loss, optimizer=optmzr, metrics=["accuracy"])
        self.lstm.load_weights(WEIGHT_PATH)

    def predict(self, data_seq, verbose=0):
        """Make a single prediction for a future prediction.

        Args:
            data_seq (np.array): Array containing temporal position data.
            verbose =0 (int): 0 = No console output. 1 = Keras console output
        Returns:
            Array containing the predicted position.

        """
        data_seq = np.expand_dims(data_seq, 0)
        prediction = self.lstm.predict(data_seq, verbose=verbose, batch_size=1)
        return prediction

    def predict_recursivly(self, data_seq, rounds, verbose=0):
        """Recursivly make a prediction on an input set.

        This function predicts multiple futures positions in sequence by
        recursivly running the network on the input sequence buffer. Each
        prediction is appened to the end of this buffer and the head removed
        with every subsiquent prediction to preserve the buffer length. In this
        way, future predictions can be forcast.
        Args:
            data_seq (np.array): Array containing temporal position data.
            rounds (int): The number of recursive predictions to be made.
            verbose =0 (int): 0 = No console output. 1 = Keras console output
        Returns:
            Array containing multiple positions predicted recursivly.

        """
        prediction_list = []
        for i in range(rounds):
            prediction = self.predict(data_seq, verbose)
            data_seq = np.roll(data_seq, -1, axis=0)
            data_seq[-1] = prediction[0]
            prediction_list.append(prediction[0])
        return prediction_list


# if __name__ == "__main__":
#     test = PredictionWrapper()
#     test_i = np.array([[0.59307359, 0.2238806], [0.5974026,  0.21641791],
#                         [0.6017316, 0.21641791]])
#     pred_list = test.predict_recursivly(test_i, 5, 1)
#     with open("uscs_peds_scaler_25000.obj", "rb") as open_file:
#         scalr = pickle.load(open_file)
#     for og in test_i:
#         print(scalr.inverse_transform([og]))
#     print()
#     for pred in pred_list:
#         print(scalr.inverse_transform([pred]))
