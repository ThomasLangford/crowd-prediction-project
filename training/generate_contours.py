"""Generate tracks using the raw mask rcnn contours."""

# Import python libraries
# import cv2
from segmentation.segmentation import MaskInterface
from karlman_filter.tracker import Tracker
import numpy as np
import os
from .training_utilities import get_jpg_list


INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"


def tracking(image_path_list, abs_file_name, segmentor):
    """Track multiple objects and save their positions to a numpy files.

    args:
        todo
    returns
    """
    cont_list = []
    count = 0
    for image_path in image_path_list:
        contours = segmentor.segment_image(image_path)
        cont_list.append(contours)
        print(count)
        count += 1
    to_save = np.asarray(cont_list)
    np.save(abs_file_name, to_save)


def track_all_in_folder(inputParentFolder, outputParentFolder):
    """DocString."""
    segmentor = MaskInterface()
    for subdir in next(os.walk(inputParentFolder))[1]:
        input_folder = inputParentFolder+"/"+subdir
        image_list = get_jpg_list(input_folder)
        abs_image_list = [os.path.join(input_folder, i) for i in image_list]
        outputFolder = outputParentFolder
        outputName = outputParentFolder+"/"+subdir+".npy"
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)
        tracking(abs_image_list, outputName, segmentor)
        print("Folder: "+subdir)


if __name__ == "__main__":
    # execute main
    track_all_in_folder(INPUT_PATH, OUTPUT_PATH)
