"""General utilities for creating global prediction data."""
import os


def get_png_list(location):
    """DocString."""
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".png")):
            fList.remove(fItem)
    return fList


def get_jpg_list(location):
    """DocString."""
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".jpg")):
            fList.remove(fItem)
    return fList


def get_image_list(location, max_images):
    """Get absolute path for all images in a folder."""
    f_list = os.listdir(location)
    count = 0
    for f_item in f_list[:]:
        if not(f_item.endswith(".jpg")):
            f_list.remove(f_item)
            count -= 1
        count += 1
        if count == max_images:
            break
    f_list = f_list[:count]
    abs_paths = [os.path.join(location, i) for i in f_list]
    return abs_paths


def get_npy_list(location, max_images):
    """Get absolute path for all npy files in a folder."""
    f_list = os.listdir(location)
    count = 0
    for f_item in f_list[:]:
        if not(f_item.endswith(".npy")):
            f_list.remove(f_item)
            count -= 1
        count += 1
        if count == max_images:
            break
    f_list = f_list[:count]
    abs_paths = [os.path.join(location, i) for i in f_list]
    return abs_paths


def get_nested_image_list(parent_location, max_images):
    """Get list of image paths."""
    count = max_images
    path_list = []
    for subdir in next(os.walk(parent_location))[1]:
        location = os.path.join(parent_location, subdir)
        subdir_list = get_image_list(location, count)
        path_list.extend(subdir_list)
        count -= len(subdir_list)
        if count <= 0:
            break
    if count > 0:
        print("Could not find ", max_images, ". But found ", len(path_list))
    assert len(path_list) == max_images
    return path_list
