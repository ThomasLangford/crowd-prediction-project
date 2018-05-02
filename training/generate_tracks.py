"""Generate tracks using the raw Mask R-CNN contours."""

# Import python libraries
# import cv2
from karlman_filter.tracker import Tracker
import numpy as np
import os
from .training_utilities import get_npy_list


IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"
OUT_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours_tracks"


def tracking(cont_path_list, out_path):
    """Apply a Kalman filter to a series of track information before saving.

    A Kalman Filter is used to extract the tracks from a single contour file.
    At the end of the sequence, the tracks are saved as a npy file for
    later processing.
    Args:
        image_path_list (list): List of absolute paths to a sequence of images.
        abs_file_name (String): Path to save the numpy file to.
    Returns:
        None

    """
    tracker = Tracker(10, 0, 0)
    count = 0
    contours_list = np.load(cont_path_list)
    for contours in contours_list:
        centers = []
        x_list = []
        y_list = []

        for contour in contours:
            # Find center of mass of contours
            x_list = [vertex[0] for vertex in contour]
            y_list = [vertex[1] for vertex in contour]
            n_vertex = len(contour)
            x = sum(x_list) / n_vertex
            y = sum(y_list) / n_vertex
            centers.append([x, y])

        if (len(centers) > 0):
            tracker.update(centers)
        # print(count)
        count += 1

    to_save = np.zeros((len(tracker.track_history), 2), dtype=object)
    for i in range(len(tracker.track_history)):
        history = []
        for j in range(len(tracker.track_history[i].center_history)):
            coords = []
            # x and y are flipped becuase of graphics geometry.
            coords.append(
                        int(tracker.track_history[i].center_history[j][0][0]))
            coords.append(
                        int(tracker.track_history[i].center_history[j][0][1]))
            history.append(coords)
        to_save[i][0] = tracker.track_history[i].frame_count
        to_save[i][1] = history
    np.save(out_path, to_save)


def track_all_in_folder(input_folder, output_folder):
    """Run tracking on all contour files in a directory.

    For each numpy file in a folder containing contours which represent the
    segmentation of pedestrians, run Kalman Tracking on each and save the
    results individually, ensuring that paths are for distint pedestrians.
    Args:
        input_folder (String): Path to the input folder
        output_folder (String): Absolute path to the output folder.
    Returns:
        None

    """
    npy_list = get_npy_list(input_folder, -1)
    # abs_image_list = [os.path.join(input_folder, i) for i in image_list]
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    for npy_path in npy_list:
        name = os.path.basename(npy_path)
        outputName = os.path.join(output_folder, name)
        print(outputName)
        tracking(npy_path, outputName)
        print("Done: ", name)


if __name__ == "__main__":
    # execute main
    track_all_in_folder(IN_DIR, OUT_DIR)
