import numpy as np
from os import listdir


def load_RD_detected_imgs(base_path, cube,  dirs, dir_labels):
    """
    Function is loading all data (files) specified with base_path, cube and dirs.
    It will look into base_path folder and then into every folder specified in list dirs.
    There it will load all files that start with string cube (Naming convention).
    Data will be labeled based on list dir_labels. Every directory in dirs has a
    corresponding label in dir_labels

    :param base_path: path to base folder
    :param cube: string specifying which cube to use (only files of that cube are loaded)
    :param dirs: list of strings representing folders in base_path directory
    :param dir_labels:list of labels for every subfolder in dirs
    :return: imgs - data loaded from file as numpy array
             positions - indices of rows and columns of maximum amplitudes in range doppler graph as numpy array
             labels - labels for each image (file)
    """

    imgs = np.array([])
    positions = np.array([])
    labels = np.array([])
    for i in range(len(dirs)):
        directory = base_path + dirs[i]

        num_of_files = 0
        for file in listdir(directory):
            if not file.startswith(cube):
                continue
            num_of_files += 1
            with open(directory+'/'+file, 'r') as f:
                row_max = float(f.readline().rstrip())
                col_max = float(f.readline().rstrip())
                rd = np.loadtxt(f, delimiter=',')

                imgs = np.append(imgs, rd)
                positions = np.append(positions, np.array([row_max, col_max]))

        file_labels = np.repeat(dir_labels[i], num_of_files)
        labels = np.append(labels, file_labels)

    return imgs, positions, labels
