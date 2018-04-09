"""
Interface for the Mask_RCNN model.

todo
"""

import os
# import sys
# import random
# import math
import numpy as np
from skimage.io import imread
from skimage.measure import find_contours

from mask_rcnn import coco
# from Mask_RCNN import utils
from mask_rcnn import model as modellib

# from. segmentation_config import InferenceConfig

# from keras.backend.tensorflow_backend import set_session


class InferenceConfig(coco.CocoConfig):
    """Static class to specify the model paramaters for MASK RCNNself.

    GPU_COUNT - Number of GPUs to use.
    IMAGES_PER_GPU - Batch size.
    NUM_CLASSES - Number of classes in model.
    """

    NAME = "Segmentation Config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # Model trained on Coco dataset.


class MaskInterface:
    """Segmentation interface for Mask RCNN."""

    def __init__(self):
        """Create a new instance of the MaskInterface class.

        ToDo
        """
        # Load Mask_RCNN
        config = InferenceConfig()
        coco_path = os.path.join(os.getcwd(), "Mask_RCNN", "mask_rcnn_coco.h5")
        self.model = modellib.MaskRCNN(mode="inference", config=config,
                                       model_dir="./logs")
        self.model.load_weights(coco_path, by_name=True)

    def get_contours(self, boxes, masks, class_ids):
        """Find contours using MaskRCNN.

        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        """
        all_contours = []

        for i in range(boxes.shape[0]):
            if class_ids[i] != 1:
                continue
            mask = masks[:, :, i]
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            # Subtract the padding and flip (y, x) to (x, y)
            contours = [np.fliplr(verts) - 1 for verts in contours]
            all_contours.append(contours[0])
        return all_contours

    def segment_image(self, abs_image_path):
        """Find contor segmentation map of image."""
        image = imread(abs_image_path)
        results = self.model.detect([image], verbose=0)[0]
        contours = self.get_contours(results['rois'], results['masks'],
                                     results['class_ids'])
        return contours
