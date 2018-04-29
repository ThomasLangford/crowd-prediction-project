"""Online display of tracks."""

from segmentation.segmentation import MaskInterface
from karlman_filter.tracker import Tracker
import numpy as np
import pickle
import os
import cv2
from math import hypot
from training.training_utilities import get_jpg_list
from prediction.prediction_wrapper import PredictionWrapper


def plot_tracks(track, image, threshold):
    """Plot tracks to image."""
    pred_color = (0, 204, 0)
    medium = (0, 128, 255)
    bad = (0, 0, 255)
    actual = (255, 0, 0)

    if len(track.prediction_list) != 0:
        point_one = track.center_history[-1][0]
        point_two = track.prediction_list[track.prediction_count]
        distance = hypot(point_one[0]-point_two[0],
                         point_one[1] - point_one[1])
        if distance > threshold * 2:
            pred_color = bad
        elif distance > threshold:
            pred_color = medium

        for point in track.prediction_list:
            cv2.circle(image, (int(point[0]), int(point[1])), 2, pred_color,
                       -1)

    if len(track.center_history) != 0:
        for point in track.center_history:
            cv2.circle(image, (int(point[0][0]), int(point[0][1])), 1,
                       actual, -1)
    return image


def reverse_transform(prediction_list, scalar):
    """Undo multiple scaled transformations."""
    return [scalar.inverse_transform([x])[0] for x in prediction_list]


def online(input_folder, interval, buffer_len, pred_len, threshold):
    """Diplay predictions online."""
    segmentor = MaskInterface()
    tracker = Tracker(10, 0)
    predicter = PredictionWrapper()
    with open("./uscs_peds_model/uscs_peds_scaler_25000.obj",
              "rb") as open_file:
        scalar = pickle.load(open_file)

    image_list = get_jpg_list(input_folder)
    abs_image_list = [os.path.join(input_folder, i) for i in image_list]

    for count, image_path in enumerate(abs_image_list):
        centers = []
        x_list = []
        y_list = []

        contours = segmentor.segment_image(image_path)
        for contour in contours:
            # Find center of mass of contours
            x_list = [vertex[0] for vertex in contour]
            y_list = [vertex[1] for vertex in contour]
            n_vertex = len(contour)
            x = int(sum(x_list) / n_vertex)
            y = int(sum(y_list) / n_vertex)
            centers.append([x, y])

        if (len(centers) > 0):
            tracker.Update(centers)

        image = cv2.imread(image_path)
        for track in tracker.tracks:

            if len(track.center_history) == 0:
                continue
            scaled = scalar.transform([track.center_history[-1][0]])[0]
            if count % interval == 0:
                print("here1")
                if len(track.prediction_buffer) < buffer_len:
                    track.prediction_buffer.append(scaled)
                    if len(track.prediction_buffer) == buffer_len:
                        track.prediction_count = pred_len
                else:
                    track.prediction_buffer[:-1] = track.prediction_buffer[1:]
                    track.prediction_buffer[-1] = scaled

                if len(track.prediction_buffer) == buffer_len:
                    if track.prediction_count == pred_len:
                        track.prediction_count = -1
                        predictions = predicter.predict_recursivly(
                            track.prediction_buffer, pred_len, 0)
                        track.prediction_list = reverse_transform(predictions,
                                                                  scalar)
                    image = plot_tracks(track, image, threshold)
                    track.prediction_count += 1
                else:
                    image = plot_tracks(track, image, threshold)
            else:
                print("here2")
                if len(track.prediction_buffer) == buffer_len:
                    track.prediction_buffer[:-1] = track.prediction_buffer[1:]
                    track.prediction_buffer[-1] = scaled
                image = plot_tracks(track, image, threshold)
        cv2.imshow("Prediction", image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
