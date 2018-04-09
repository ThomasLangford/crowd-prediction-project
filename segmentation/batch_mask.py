import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))

# import ../Mask_RCNN.coco
from mask_rcnn import coco
from mask_rcnn import utils
from mask_rcnn import model as modellib
from mask_rcnn import visualize


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "batck_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "Mask_RCNN", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


IN_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/ucsdpeds/vidf_jpg"
OUT_DIR = "C:/Users/Nedsh/Documents/CS/Project/DatasetRaw/mask_ucsdpeds"


def getJPGList(location):
    """DocString."""
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".jpg")):
            fList.remove(fItem)
    return fList


def massImageMask(inputParentFolder, outputParentFolder):
    """DocString."""
    # count = 0
    for subdir in next(os.walk(inputParentFolder))[1]:
        input_folder = inputParentFolder+"/"+subdir
        output_folder = outputParentFolder+"/"+subdir
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        for imageName in getJPGList(input_folder):
            image = skimage.io.imread(os.path.join(input_folder, imageName))
            # Run detection
            results = model.detect([image], verbose=0)
            # Visualize results
            r = results[0]
            mask = visualize.get_instances(image, r['rois'], r['masks'],
                                           r['class_ids'], class_names,
                                           r['scores'])

            out_name = os.path.splitext(imageName)[0]
            out_path = os.path.join(output_folder, out_name)
            skimage.io.imsave(out_path+".jpg", mask)
            print(out_name)

        print("Folder: "+subdir)


if __name__ == "__main__":
    massImageMask(IN_DIR, OUT_DIR)
