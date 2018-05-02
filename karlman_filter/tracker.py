"""Tracker class for the Kalman filter.

This module contains the tracker class which controls the operation of
the Kalman filter during normal tracking operations. Additionally, the
class creates and manages the assigned tracks and relates them back to
the centroids.
"""
import numpy as np
from math import hypot
from .track import Track
from scipy.optimize import linear_sum_assignment


class Tracker:
    """Tracker class for the Kalman filter.

    This class controls the operation of the Kalman filter during normal
    tracking operations. Additionally, the class creates and manages the
    individual track data objects, removing them if the associated pedestrian
    has not been detected for a period of time. The tracker uses the Hungarian
    algorithm to assign predicions to the closes centroid in an image.
    Attributes:
        track_count (int): The number of tracks currently active
        dist_thresh (float): The maximum score for a track to be assigned to a
                                centroid.
        max_frames_to_skip (int): The maximum number of frames a pedestrian
                                    can go undetedted before deletion.
        tracks (list): List of currently active tracks.
        track_history (list): List of all track objects created by the tracker.

    """

    def __init__(self, dist_thresh, max_frames_to_skip):
        """Initalise a new instanse of the Tracker class.

        This new instance does not include any tracks at the time of its
        instantiation. Instead, to generate tracks the update function must
        be first run on a set of centroids.
        Args:
            dist_thresh (float): The maximum score for a track to be assigned
                                    to a centroid.
            max_frames_to_skip (int): The maximum number of frames a pedestrian
                                        can go undetedted before deletion.

        """
        self.track_count = 0
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.track_history = []

    def update(self, centroids):
        """Update the state of the tracks based on observed centroids.

        This procedure controls the lifecycle of the Kalman filters and the
        associated tracking objects. The filters are assigned to observed
        centroid in the current frame by calculating the difference between
        the current observed position and the predicion made on the current
        frame. To calculate these assignments efficiently, a Hungarian
        Algorithm is used to assign the predictions bassed off of a cost
        matrix. The function could be split into smaller sub functions but
        that would be at the cost of increased computation time due to the
        amount of prior information required by each step.
        Args:
            centroids (list): A list of observed pedestrian center points.
        Returns:
            None

        """
        # Create tracks if there are detections and no tracks in the currently
        if len(self.tracks) == 0:
            for i in range(len(centroids)):
                track = Track(centroids[i], self.track_count)
                self.track_count += 1
                self.tracks.append(track)
                self.track_history.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(centroids)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(centroids)):
                try:
                    diff = self.tracks[i].prediction - centroids[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except Exception:
                    pass

        # Calculate the squared error of the cost matrix
        cost = (0.5) * cost
        # Using Hungarian Algorithm to assgin predicitons to centroids
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Find all tracks without a valid assignment
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                track = self.tracks[i]
                if len(track.center_history) > 1:
                    coords = track.center_history[-1][0]
                    dist = hypot(centroids[assignment[i]][0] - coords[0],
                                 centroids[assignment[i]][1] - coords[1])
                else:
                    dist = -1
                # Check against cost and the distance between the last known
                # point and the assigned centroid
                if cost[i][assignment[i]] > self.dist_thresh or dist > 10:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                    self.tracks[i].skipped_frames += 1
            else:
                self.tracks[i].skipped_frames += 1

        # Delete filters which have not had a corresponding centroid assigned
        # to them within a certain threshold.
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)

        if len(del_tracks) > 0:
            copy_tracks = []
            for i in range(len(self.tracks)):
                if i not in del_tracks:
                    copy_tracks.append(self.tracks[i])
            self.tracks = copy_tracks
            copy_ass = []
            for i in range(len(assignment)):
                if i not in del_tracks:
                    copy_ass.append(assignment[i])
            assignment = copy_ass
            copy_unass = []
            for i in range(len(un_assigned_tracks)):
                if i not in del_tracks:
                    copy_unass.append(un_assigned_tracks[i])
            un_assigned_tracks = copy_unass

        # List all unassigned centroids
        un_assigned_detects = []
        for i in range(len(centroids)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks for unassigned detects as long as there are not
        # other free tracks. This is added to avoid an error cascade as
        # discussed in the report.
        if len(un_assigned_detects) != 0 and len(un_assigned_tracks) == 0:
            for i in range(len(un_assigned_detects)):
                track = Track(centroids[un_assigned_detects[i]],
                              self.track_count)
                track.center_history.append(
                    [centroids[un_assigned_detects[i]]])
                track.frame_count += 1
                self.track_count += 1
                self.tracks.append(track)
                self.track_history.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            centroids[assignment[i]], 1)
                self.tracks[i].center_history.append(
                                            [centroids[assignment[i]]])
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)
                self.tracks[i].center_history.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
            self.tracks[i].frame_count += 1
