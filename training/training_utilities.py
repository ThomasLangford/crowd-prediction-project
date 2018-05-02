"""General utilities for creating prediction data for the LSTM network."""
import os


def get_png_list(location):
    """Get the names of all png files in a location.

    Args:
        location (String): Path to folder.
    Returns:
        A list of all the png files in the folder.

    """
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".png")):
            fList.remove(fItem)
    return fList


def get_jpg_list(location):
    """Get the names of all jpg files in a location.

    Args:
        location (String): Path to folder.
    Returns:
        A list of all the jpg files in the folder.

    """
    fList = os.listdir(location)
    for fItem in fList[:]:
        if not(fItem.endswith(".jpg")):
            fList.remove(fItem)
    return fList


def get_image_list(location, max_images):
    """Get the absolute file path to all jpg images in a folder.

    Args:
        location (String): Path to folder.
        max_images (int): The number of images to select. -1 selects all.
    Returns:
        A list with the absolute path to all jpg images in a folder.

    """
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
    """Get the absolute file path to all npy files in a folder.

    Args:
        location (String): Path to folder.
        max_images (int): The number of images to select. -1 selects all.
    Returns:
        A list with the absolute path to all .npy files in a folder.

    """
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
    """Get the absolute file path to all jpg images within subdirecties.

    Args:
        location (String): Path to parent folder folder.
        max_images (int): The number of images to select. -1 selects all.
    Returns:
        A list with the absolute path to all jpg images in a sub directories.

    """
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
