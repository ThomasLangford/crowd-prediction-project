"""Estimate the future state of pedestrian motion useing a Kalman Filter.

This file contains the class, KalmanFilter, used to store the state of the
modeled system to estimate the state in a future timestep. The Kalman Filter
is applied as part of the project to the modelling of human motion, relating
the predicted positions back to detected centroids, allowing for people to be
tracked between frames.
"""
import numpy as np


class KalmanFilter:
    """Estimate the future state of a system.

    This class is used to store the state of the modeled system to estimate the
    state in a future timestep. The Kalman Filter is applied as part of the
    project to the modelling of human motion, relating the predicted positions
    back to detected centroids, allowing for people to be tracked between
    frames. In this application, each instance of the Kalman filter has to
    be allocated to only one pedestrian as that pedestrians 'state' differs
    from those around it as the differnces in velocities will produce erronious
    results.
    Attributes:
        dt (float): Delta time, used to create state transition model.
        self.A (list): The observation model.
        self.x (array): Vector of previous system state (position).
        self.b (array): Vector of the current observed state.
        self.P (array): Vector of previous system error.
        self.F (array): State transition model.
        self.Q (array): Covariance of the process noise.
        Self.R (array): Covariance of the observation noise.
        Self.lastResult (array): Last predicted position.

    """

    def __init__(self):
        """Create a new instance of the Kalman Filter.

        This constuctor initalises the inital state of the filter as well as
        many of the variables within it such as the state transtion model. The
        variables initalised here are used in the subsiquent prediction and
        updating steps.
        Args:
            None

        """
        self.dt = 0.005

        self.A = np.array([[1, 0], [0, 1]])
        self.x = np.zeros((2, 1))

        self.b = np.array([[0], [255]])

        self.P = np.diag((3.0, 3.0))
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])

        self.Q = np.eye(self.x.shape[0])
        self.R = np.eye(self.b.shape[0])

        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict the state vector and variance of uncertainty.

        This function uses the Time Update prediction equation from the two
        part Kalman Filter equations to predict the position of the pedestrian
        in the next frame. It first predicts the state of the system in the
        next time step and then predicts the expected covariance of system
        error.
        Args:
            None
        Returns:
            The predicted state vector.

        """
        # Predicted state estimate
        self.x = np.round(np.dot(self.F, self.x))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.x
        return self.x

    def correct(self, b, allocated):
        """Update the state vector and uncertainty using an observation.

        This function uses the Measurement Update correction equation from the
        two part Kalman filter equations to correct the state of the system
        and the position of the pedestrian in the next frame. If no observation
        is made due to either noise in the system or the detection not being
        made in the case of the pedestrian centroid then the last predicted
        position will be used to correct the state and covariance instead.
        Args:
            b (array): Vector of observations.
            allocated (bool): True if b contains a valid observation, false
                                if not.
        Returns:
            The updated state vector.

        """
        if not allocated:
            # Use the prediction instead
            self.b = self.lastResult
        else:
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.x = np.round(self.x + np.dot(K, (self.b - np.dot(self.A,
                                                              self.x))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.x
        return self.x
