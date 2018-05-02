"""Track data object to store the state of the system local to a pedestiran.

This file contains the data container Track class, used to store the history
of that tracks positions as well as the state of the Kalman filter. It has
been sperated out from ease of maintanance as methods or more attributes
may be added at a later stage of development.
"""

import numpy as np
from .kalman_filter import KalmanFilter


class Track:
    """Track data object to store the state of the system local to a pedestiran.

    This class acts a data store for the Tracker to hold information about
    the state of the Kalman Filter as well as to keep a track of previous
    positions that a pedestrian has occupied. The class also contains
    other metrics to facilitate the generation of datasets as well as analysis
    on the quality of the tracking.
    Attributes:
        track_id (int): ID number of the track.
        KF (KalmanFilter): Kalman filter state object.
        prediction (array): Last predicted position.
        skipped_frames (int): Number of frames without a centroid allocated.
        color (tripple): RGB colour for highlighting the track in a frame.
        frame_count (int): Number of frames in the track.
        center_history (list): List of previously occupied positions.
        prediction_buffer(list): Prediction LSTM input buffer.
        prediction_list (list): A number of future predicted positions.
        prediction_count (int: The number of frames the prediction has left.

    """

    def __init__(self, position, track_id):
        """Initalise a new track object to store the track state.

        This class is a data container for the Tracker class, used to store the
        history of a single track as well as the state of the associated Kalman
        filter.
        Args:
            position (array): Inital position for the track to be created on.
            track_id (int): Unique identifier for an active track.
        Return:
            None
        """
        self.track_id = track_id

        # Set up the Kalman filter state
        self.KF = KalmanFilter()
        self.prediction = np.asarray(position)
        self.KF.lastResult = self.prediction
        for _ in range(5):
            self.prediction = self.KF.correct(position, 1)
        self.KF.lastResult = self.prediction
        self.skipped_frames = 0

        self.color = list(np.random.random(size=3) * 256)

        # Set prediction and display variables
        self.frame_count = 0
        self.center_history = []
        self.prediction_buffer = []
        self.prediction_list = []
        self.prediction_count = 0
