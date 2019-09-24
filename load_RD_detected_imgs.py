import numpy as np
from os import listdir


def load_RD_detected_imgs(base_path, cube,  dirs, dir_labels):
    imgs = np.array([])
    positions = np.array([])
    labels = np.array([])
    for i in range(len(dirs)):
        directory = base_path + dirs[i]
        file_labels = np.repeat(dir_labels[i], len(listdir(directory)))
        labels = np.append(labels, file_labels)

        for file in listdir(directory):
            if not file.startswith(cube):
                continue
            with open(directory+'/'+file, 'r') as f:
                row_max = float(f.readline().rstrip())
                col_max = float(f.readline().rstrip())
                rd = np.loadtxt(f, delimiter=',')

                imgs = np.append(imgs, rd)
                positions = np.append(positions, np.array([row_max, col_max]))

    return imgs, positions, labels
