"""This is the main program coupling the seperate sections of the project.

The script creates new instances of each portion of the LSTM prediction model
to allow the model to be run in real time on an input sequence. It accesses
the segmentation capabilities of the Mask R-CNN network through the
MaskInterface class to generate contours. From the contours, the pedestrians
can be tracked through the frame using the Kalman filter and associated
tracking class. The prediction buffer for each is then filled and then passed
to the prediction LSTM network, with the results of this network run back on
the screen.

To run this file, run the 'run_prediction.py' file in the parent directory.

"""

from segmentation.segmentation import MaskInterface
from karlman_filter.tracker import Tracker
import numpy as np
import pickle
import os
import cv2
from math import hypot, ceil
from training.training_utilities import get_jpg_list
from prediction.prediction_wrapper import PredictionWrapper


def plot_tracks(track, image, threshold):
    """Display the tracks of each detected pedestrian along any predictions.

    This function takes a track object and displayes the path the pedestrian
    has taken in previous frames and also the predicted future positions of
    that pedestrian if there is one avalible. The color of the predicted path
    depends on the degree at which the path is differs from the prediction at
    the relevent time step.
    Args:
        track (Track): Track object containing information about the positions.
        image (CV2.image): Image canvas object to be drawn onto.
        threshold (float): The max distance between predicion and observation
    Returns:
        The updated image canvas.
    """
    pred_color = (0, 204, 0)
    medium = (0, 128, 255)
    bad = (0, 0, 255)

    if len(track.center_history) > 1:
        for i in range(len(track.center_history) - 1):
            point_1 = track.center_history[i][0]
            point_1_tup = (int(point_1[0]), int(point_1[1]))
            point_2 = track.center_history[i+1][0]
            point_2_tup = (int(point_2[0]), int(point_2[1]))
            cv2.line(image, point_1_tup, point_2_tup, track.color, 2)

    if len(track.prediction_list) != 0:
        point_one = track.center_history[-1][0]
        point_two = track.prediction_list[track.prediction_count]
        distance = hypot(point_one[0]-point_two[0],
                         point_one[1] - point_one[1])
        print(distance)
        if distance > threshold * 2:
            pred_color = bad
        elif distance > threshold:
            pred_color = medium

        for i in range(len(track.prediction_list) - 1):
            point_1 = track.prediction_list[i]
            point_1_tup = (ceil(point_1[0]), ceil(point_1[1]))
            point_2 = track.prediction_list[i+1]
            point_2_tup = (ceil(point_2[0]), ceil(point_2[1]))
            cv2.line(image, point_1_tup, point_2_tup, pred_color, 2)

    return image


def reverse_transform(prediction_list, scalar):
    """Inverse transform each element in a list.

    Args:
        prediction_list (array): List of predictions to be transformed.
        scalar (Scikit Scalar): Trained MinMax scalar.
    Returns:
        The input list with the values inverted by the scalar.

    """
    return [scalar.inverse_transform([x])[0] for x in prediction_list]


def online(input_folder, interval, buffer_len, pred_len, threshold):
    """Combine the project modules in an online demonstration.

    The script creates new instances of each portion of the LSTM prediction
    model to allow the model to be run in real time on an input sequence.
    It accesses the segmentation capabilities of the Mask R-CNN network through
    the MaskInterface class to generate contours. From the contours, the
    pedestrians can be tracked through the frame using the Kalman filter and
    associated tracking class. The prediction buffer for each is then filled
    and then passed to the prediction LSTM network, with the results of this
    network run back on the screen.
    Args:
        input_folder (String): Path to the image sequence folder.
        interval (int): Gap between frames to be used for the prediciton.
        buffer_len (int): Length of the input for the LSTM network.
        pred_len (int): Number of predictions to be made.
        threshold (float): When to change the prediction colour.
    Returns:
        None

    """
    segmentor = MaskInterface()
    # tracker = Tracker(10, 0)
    tracker = Tracker(5, 2)
    predicter = PredictionWrapper()
    with open("./model/uscs_peds_scaler_20000.obj",
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
            cnt = np.array(contour, dtype=np.int32)
            cnt = cnt.reshape((-1, 1, 2))
            _, radius = cv2.minEnclosingCircle(cnt)

            if radius < 10 or radius > 30:
                continue

            # Find center of mass of contours
            x_list = [vertex[0] for vertex in contour]
            y_list = [vertex[1] for vertex in contour]
            n_vertex = len(contour)
            x = int(sum(x_list) / n_vertex)
            y = int(sum(y_list) / n_vertex)
            centers.append([x, y])

        if (len(centers) > 0):
            tracker.update(centers)

        image = cv2.imread(image_path)

        for track in tracker.tracks:
            if len(track.center_history) == 0:
                continue
            scaled = scalar.transform([track.center_history[-1][0]])[0]
            if count % interval == 0:
                if len(track.prediction_buffer) < buffer_len:
                    track.prediction_buffer.append(scaled)
                    if len(track.prediction_buffer) == buffer_len:
                        track.prediction_count = pred_len
                else:
                    track.prediction_buffer[:-1] = track.prediction_buffer[1:]
                    track.prediction_buffer[-1] = scaled

                if len(track.prediction_buffer) == buffer_len:
                    if track.prediction_count == pred_len:
                        track.prediction_count = 0
                        predictions = predicter.predict_recursivly(
                            track.prediction_buffer, pred_len, 0)
                        track.prediction_list = reverse_transform(predictions,
                                                                  scalar)
                    image = plot_tracks(track, image, threshold)
                    track.prediction_count += 1
                else:
                    image = plot_tracks(track, image, threshold)
            else:
                if len(track.prediction_buffer) == buffer_len:
                    track.prediction_buffer[:-1] = track.prediction_buffer[1:]
                    track.prediction_buffer[-1] = scaled
                image = plot_tracks(track, image, threshold)

        cv2.imshow("Prediction", image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
