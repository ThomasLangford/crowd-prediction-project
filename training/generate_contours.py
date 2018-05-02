"""Generate numpy files containing raw contour files.

Due to the time taken for Mask R-CNN to iterate over the entire dataset,
the best method for generating the dataset is to generate the raw contours
first and then apply the filter onto it afterwards. This ensure that if changes
are made to the Kalman filter, you do not have to wait for the entire dataset
to be run again.
"""

from segmentation.segmentation import MaskInterface
import numpy as np
import os
from .training_utilities import get_jpg_list


INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"
OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours"


def tracking(image_path_list, abs_file_name, segmentor):
    """Segment each image in a series and then save to file.

    This function segments each image individually using the semgentation
    wrapper class to abstract away the network functionality. At the end of
    the sequence, the contours are saved as a npy file for later processing.
    Args:
        image_path_list (list): List of absolute paths to a sequence of images.
        abs_file_name (String): Path to save the numpy file to.
        segmentor (MaskInterface): Interface object for the Mask R-CNN model.
    Returns:
        None

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


def track_all_in_folder(input_parent_folder, output_folder):
    """Segments all images within sub directories and saves the result.

    For each set of images in a subdirectory, this function applies the Mask
    R-CNN segmentation model to each sequence  to generate a set of contours.
    The contour files are saved individually per sequence as .npy files in the
    output folder specidied.
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
