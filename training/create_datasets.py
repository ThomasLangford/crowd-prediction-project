"""Create positinal datasets from a set of contour files.

These scripts are used to assemble sets of tracked information at a specified
frame interval and length. Additionally, a number of frames can be skipped
before the dataset is compiled to allow for the creation of a seperate file
containing unseen data.

Example:
    python create_datasets.py
"""

from _mask_utilities import get_npy_list
from sklearn.preprocessing import MinMaxScaler
# from random import shuffle
import numpy as np
import pickle
import os

# INPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/position_contours_tracks"
# OUTPUT_PATH = "C:/Users/Nedsh/Documents/CS/Project/DatasetLabelled/local_tracking_bis"


def create_data(n_samples, timesteps, interval, input_path, output_path_data,
                output_path_label, after=0):
    """Create a seperate track data and label file.

    This function generates a .npy file containing a set of positions with a
    set frame interval between them to be used as the data file in the training
    of the predition LSTM network. Additionally, a seperate label file will be
    contain the next position after the corresponding data list with the same
    interval as before. It will collate the dataset from a list of position
    data extracted from a video sequence which is stored in multiple .npy
    files. This function will also save the MinMax scalar in the same directory
    as the data file.

    Args:
        n_samples (int): Number of samples to include in the dataset.
        timesteps (int): Number of positions in a single sample.
        interval (int): Number of frames between each position 1 = every frame.
        input_path (String): Location of the individual .npy position files.
        output_path_data (String): Path, including name, for the data file.
        output_path_label (String): Path, including name, for the lbl file.
        after =0 (int): Number of samples to skip before compiling the data.
    Returns:
        None

    """
    # Determine the minimum number of positions in a track
    treshhold = (timesteps + 1) * interval
    # Determine the individual positions to find
    n_fetch = (n_samples * interval) + timesteps + 1 + after
    pos_list = np.zeros((0, 2))

    # Load all files into memory
    for path in get_npy_list(input_path, n_fetch):
        pos_arr = np.load(path)
        pos_list = np.concatenate((pos_list, pos_arr))

    # Calculate the number of valid paths
    valid_paths = []
    for arr in pos_list:
        if(arr[0] > treshhold):
            for i in range(0, len(arr[1]), treshhold):
                path = arr[1][i:i+treshhold:interval]
                if len(path) == timesteps + 1:
                    valid_paths.append(path)

    if len(valid_paths) < n_samples:
        raise ValueError("Cannot find ", n_samples, ".Found ",
                         len(valid_paths))

    print(len(valid_paths))
    raw_list = valid_paths[after: n_samples]

    # shuffle(valid_paths)
    # raw_list = np.zeros((n_samples, timesteps+1, 2))
    # for i in range(n_samples+after):
    #     if i < after:
    #         continue
    #     index = i - after
    #     raw_list[index] = valid_paths[i]

    # Use the MinMax scalar to standadise the position data in the raw list.
    # However each value needs to be passed into the scalar to allow for the
    # resultant values from the network to be untransformed.
    scaler = MinMaxScaler()
    for i in range(raw_list.shape[0]):
        for j in range(raw_list.shape[1]):
            scaler.partial_fit([raw_list[i][j]])
    for i in range(raw_list.shape[0]):
        for j in range(raw_list.shape[1]):
            raw_list[i][j] = scaler.transform([raw_list[i][j]])

    data_list = np.zeros((n_samples, timesteps, 2))
    label_list = np.zeros((n_samples, 2))

    # Place the data and label into final arrays
    for i in range(n_samples+after):
        if i < after:
            continue
        index = i - after
        path = raw_list[index]
        label_list[index] = path[-1]
        data_list[index] = path[:-1]

    # print(data_list[0])
    # print(label_list[0])
    # print(scaler.inverse_transform([label_list[0]]))
    # print(scaler.inverse_transform([data_list[0][-1]]))

    # pickle_name = '/uscs_peds_scaler_' + str(n_samples) + '_' + str(timesteps)
    pickle_name = '/scaler_' + str(n_samples) + '_' + str(timesteps)
    pickle_name = pickle_name + '_' + str(interval) + '.obj'
    pickle_name = os.path.dirname(output_path_data) + pickle_name

    # Save resultant files
    with open(pickle_name, 'wb') as file_out:
        pickle.dump(scaler, file_out)
    np.save(output_path_data, data_list)
    np.save(output_path_label, label_list)


# if __name__ == "__main__":
#     samples = 20000
#     test_samples = 500
#     frames = 3
#     # 1 is every frame, 2 is every 2nd frame.
#     interval = 2
#     folder = "/postitions_frames-" + str(frames) + "_steps-" + str(interval)
#     folder_path = OUTPUT_PATH + folder
#     train_name = "_positions_" + str(samples) + "_s_" + str(frames)
#     test_name = "_positions_" + str(test_samples) + "_s_" + str(frames)
#     name2 = "_i_" + str(interval) + "_minmax" + "_ucsdpeds.npy"
#
#     if not os.path.isdir(folder_path):
#         os.makedirs(folder_path)
#
#     data_path = folder_path + "/traindata" + train_name + name2
#     lbl_path = folder_path + "/trainlabel" + train_name + name2
#     create_data(samples, frames, interval, INPUT_PATH, data_path, lbl_path)
#
#     data_path = folder_path + "/testdata" + test_name + name2
#     lbl_path = folder_path + "/testlabel" + test_name + name2
#     create_data(test_samples, frames, interval, INPUT_PATH, data_path,
#                 lbl_path, after=samples)
