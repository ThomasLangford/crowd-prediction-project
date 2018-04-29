'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from .kalman_filter import KalmanFilter
# from common import dprint
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.KF.lastResult = self.prediction
        for _ in range(5):
            self.prediction = self.KF.correct(prediction, 1)
        self.KF.lastResult = self.prediction
        self.skipped_frames = 0  # number of frames skipped undetected
        self.frame_count = 0
        self.center_history = []
        self.color = list(np.random.random(size=3) * 256)


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.track_history = []
        self.center = []

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                self.track_history.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    print("Err!")
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                    self.tracks[i].skipped_frames += 1
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            print("Need_to_del ", len(del_tracks), " tracks.")
            for i, index in enumerate(del_tracks):
                if index < len(self.tracks):
                    # print("Deleted a track")
                    # del self.tracks[index]
                    # del assignment[index]
                    #
                    # for j in range(i, len(del_tracks)):
                    #     del_tracks[j] -= 1
                    pass
                else:
                    print("ERROR: id is greater than length of tracks 1")
                    pass
            print("Before tracks len ", len(self.tracks))
            copy_tracks = []
            for i in range(len(self.tracks)):
                if i not in del_tracks:
                    copy_tracks.append(self.tracks[i])
            self.tracks = copy_tracks
            print("After tracks len ", len(self.tracks))
            print("Before ass len ", len(self.tracks))
            copy_ass = []
            for i in range(len(assignment)):
                if i not in del_tracks:
                    copy_ass.append(assignment[i])
            assignment = copy_ass
            print("After tracks len ", len(self.tracks))

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                track.center = detections[un_assigned_detects[i]]
                track.center_history.append(track.prediction)
                track.frame_count += 1
                self.trackIdCount += 1
                self.tracks.append(track)
                self.track_history.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1)
                self.tracks[i].center = detections[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)
                self.tracks[i].center = self.tracks[i].prediction[0]

            self.tracks[i].KF.lastResult = self.tracks[i].prediction
            # add posiition to history
            self.tracks[i].center_history.append(self.tracks[i].prediction)

            self.tracks[i].frame_count += 1

        print("Assigments", len(assignment))
        print("Un Assigned Decs", len(un_assigned_detects))
        print("Un Assigned Tracks", len(un_assigned_tracks))
        print("Tracks", len(self.tracks))
