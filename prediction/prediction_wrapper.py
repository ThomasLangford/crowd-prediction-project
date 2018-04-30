"""Wrapper for the lstm network."""

# Keras Imports
import keras.models as models
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


MODEL_PATH = "./uscs_peds_model/local_positional_one_lstm_one_dense_u-128.json"
# WEIGHT_PATH = "./prediction_weights_2018-04-28.hdf5"
WEIGHT_PATH = "./uscs_peds_model/prediction_weights_2018-04-30_13_24.hdf5"


class PredictionWrapper:
    """Wrapper class for the lstm network."""

    def __init__(self):
        """Create a new wrapper instance."""
        # Create network.
        optimizer = Adam(lr=0.0001)
        loss = "mean_squared_error"
        with open(MODEL_PATH) as model_file:
            self.lstm = models.model_from_json(model_file.read())
        self.lstm .compile(loss=loss, optimizer=optimizer,
                           metrics=["accuracy"])
        self.lstm.load_weights(WEIGHT_PATH)

    def predict(self, data_seq, verbose):
        """Make a prediction."""
        data_seq = np.expand_dims(data_seq, 0)
        prediction = self.lstm.predict(data_seq, verbose=verbose, batch_size=1)
        return prediction

    def predict_recursivly(self, data_seq, rounds, verbose):
        """Recursivly make a prediction."""
        prediction_list = []
        for i in range(rounds):
            prediction = self.predict(data_seq, verbose)
            data_seq = np.roll(data_seq, -1, axis=0)
            data_seq[-1] = prediction[0]
            prediction_list.append(prediction[0])
        return prediction_list



if __name__ == "__main__":
    test = PredictionWrapper()
    test_i = np.array([[0.59307359, 0.2238806], [0.5974026,  0.21641791],
                        [0.6017316, 0.21641791]])
    # Pred [[0.6002252 0.2378054]]
    # Labl [0.60606061 0.21641791]
    # print(test.predict(test_i, 1))
    pred_list = test.predict_recursivly(test_i, 5, 1)
    with open("uscs_peds_scaler_25000.obj", "rb") as open_file:
        scalr = pickle.load(open_file)
    for og in test_i:
        print(scalr.inverse_transform([og]))
    print()
    for pred in pred_list:
        print(scalr.inverse_transform([pred]))
