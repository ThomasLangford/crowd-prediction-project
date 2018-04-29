"""
Track object.
"""

import numpy as np
from .kalman_filter import KalmanFilter


class Track:
    """Track class for every object to be trackedself.

    Attributes:
        None

    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class.

        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount
        # Create Kalman information
        self.KF = KalmanFilter()
        self.prediction = np.asarray(prediction)
        # Apply the prediction ten times to center the initial position
        # on the centroid.
        for _ in range(10):
            self.KF.correct(prediction, 1)
        self.skipped_frames = 0

        # Set prediction and display variables
        self.frame_count = 0
        self.center_history = []
        self.prediction_buffer = []
        self.prediction_list = []
        self.prediction_count = -1
