"""Generate tracks using the raw mask rcnn contours."""

# Import python libraries
# import cv2
from karlman_filter.tracker_old import Tracker
import numpy as np
import os
from .training_utilities import get_npy_list


IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"
OUT_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours_tracks"


def tracking(cont_path_list, out_path):
    """Track multiple objects and save their positions to a numpy files.

    args:
        todo
    returns
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
            tracker.Update(centers)
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


def track_all_in_folder(inputParentFolder, outputParentFolder):
    """DocString."""
    npy_list = get_npy_list(inputParentFolder, -1)
    # abs_image_list = [os.path.join(input_folder, i) for i in image_list]
    if not os.path.isdir(outputParentFolder):
        os.makedirs(outputParentFolder)
    for npy_path in npy_list:
        name = os.path.basename(npy_path)
        outputName = os.path.join(outputParentFolder, name)
        print(outputName)
        tracking(npy_path, outputName)
        print("Done: ", name)


if __name__ == "__main__":
    # execute main
    track_all_in_folder(IN_DIR, OUT_DIR)
