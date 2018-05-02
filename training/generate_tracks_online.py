"""Generate tracks using the raw mask rcnn contours."""

# Import python libraries
# import cv2
from segmentation.segmentation import MaskInterface
from karlman_filter.tracker import Tracker
import numpy as np
import os
from .training_utilities import get_jpg_list


INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/online_track_positions"


def tracking(image_path_list, abs_file_name, segmentor):
    """Segment each image in a series and apply the Kalman filter before saving.

    This function segments each image individually using the semgentation
    wrapper class to abstract away the network functionality. A Kalman Filter
    is used to extract the tracks from the series of input images. At the end
    of the sequence, the tracks are saved as a npy file for later processing.
    Args:
        image_path_list (list): List of absolute paths to a sequence of images.
        abs_file_name (String): Path to save the numpy file to.
        segmentor (MaskInterface): Interface object for the Mask R-CNN model.
    Returns:
        None

    """
    tracker = Tracker(10, 0)
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
            coords.append(
                        int(tracker.track_history[i].center_history[j][0][0]))
            coords.append(
                        int(tracker.track_history[i].center_history[j][0][1]))
            history.append(coords)
        to_save[i][0] = tracker.track_history[i].frame_count
        to_save[i][1] = history
    np.save(abs_file_name, to_save)


def track_all_in_folder(input_parent_folder, output_folder):
    """Segments all images within sub directories, apply a filter and save.

    For each set of images in a subdirectory, this function applies the Mask
    R-CNN segmentation model to each sequence  to generate a set of contours.
    On each set of contours, a Kalman filter is applied in real time to build
    up a information about paths taken by pedestirans in the frame. The track
    files are saved individually per sequence as .npy files in the output
    folder specidied.
    Args:
        input_parent_folder (String): Path to the parent folder.
        output_folder (String): Absolute path to the output folder.
    Returns:
        None

    """
    segmentor = MaskInterface()
    for subdir in next(os.walk(input_parent_folder))[1]:
        input_folder = input_parent_folder+"/"+subdir
        image_list = get_jpg_list(input_folder)
        abs_image_list = [os.path.join(input_folder, i) for i in image_list]
        outputFolder = output_folder
        outputName = output_folder+"/"+subdir+".npy"
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)
        tracking(abs_image_list, outputName, segmentor)
        print("Folder: "+subdir)


if __name__ == "__main__":
    # execute main
    track_all_in_folder(INPUT_PATH, OUTPUT_PATH)
