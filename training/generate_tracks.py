'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import cv2
from segmentation.segmentation import MaskInterface
from karlman_filter.tracker import Tracker
import numpy as np
import os
from .training_utilities import get_jpg_list


INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/online_track_positions"


def tracking(image_path_list, abs_file_name, segmentor):
    """Track multiple objects and save their positions to a numpy files.

    args:
        todo
    returns:
        None
    """
    segmentor = MaskInterface()
    tracker = Tracker(10, 0, 0)
    count = 0
    for image_path in image_path_list:
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
        print(count)
        count += 1

    to_save = np.zeros((len(tracker.track_history), 2), dtype=object)
    for i in range(len(tracker.track_history)):
        history = []
        for j in range(len(tracker.track_history[i].center_history)):
            coords = []
            # x and y are flipped becuase of graphics geometry.
            coords.append(tracker.track_history[i].center_history[j][1][0])
            coords.append(tracker.track_history[i].center_history[j][0][0])
            history.append(coords)
        to_save[i][0] = tracker.track_history[i].frame_count
        to_save[i][1] = history
    np.save(abs_file_name, to_save)


def track_all_in_folder(inputParentFolder, outputParentFolder):
    """DocString."""
    segmentor = MaskInterface()
    for subdir in next(os.walk(inputParentFolder))[1]:
        inputFolder = inputParentFolder+"/"+subdir
        image_list = get_jpg_list(inputFolder)
        abs_image_list = [os.path.join(inputFolder, i) for i in image_list]
        outputFolder = outputParentFolder
        outputName = outputParentFolder+"/"+subdir+".npy"
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)
        tracking(abs_image_list, outputName, segmentor)
        print("Folder: "+subdir)


if __name__ == "__main__":
    # execute main
    track_all_in_folder(INPUT_PATH, OUTPUT_PATH)
