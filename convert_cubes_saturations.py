import numpy as np
import os
from csv import reader
from math import floor, ceil
from load_RD_detected_imgs import load_RD_detected_imgs
from draw_saved_rectangle import draw_saved_rectangle


def load_data(file_path, data_size):
    """
    Function is loading all data from CSV file into numpy array.

    :param file_path: path to the file containing data in CSV format
                      (file cannot contain extra lines in the beginning, only CSV data)
    :param data_size: number of integers (or floats) in the file
    :return: numpy array containing all data from the file
    """
    f = open(file_path, 'r')
    nump = np.zeros((data_size))
    cs = reader(f)
    cnt = 0
    for row in cs:
        for col in row:
            nump[cnt] = float(col)
            cnt += 1

    f.close()
    return nump


def convert_cubes(base_path, end_dir, source_dir, from_cube, to_cube, to_cube_scalar, rows, columns, rect_rows, rect_cols, saturation=True):
    """
    Function takes already classified and saved rectangle files, takes the position of rectangles from those files, and
    generates new rectangle files with same position, but using the data from another cube. Also while processing data
    it is possible to choose whether or not to do saturation or scale each rectangle separately.
    Once you've labeled the data, you can use this function to generate data with different kind of processing.
    For example you can convert data of one cube with saturation to the same cube (same unlabeled range doppler),
    but without saturation.

    :param base_path: path to the labeled data where positions of rectangles will be loaded
    :param end_dir: path to the directory where new data will be generated
    :param source_dir: path to the directory where range doppler graphs are stored
    :param from_cube: cube of labeled data from which positions of rectangles will be loaded
    :param to_cube: cube of range doppler data that will be taken to generate new data
    :param to_cube_scalar: number that scales all data in every loaded range doppler graph
    :param rows: number of rows in range doppler graph
    :param columns: number of columns in range doppler graph
    :param rect_rows: number of rows in rectangles
    :param rect_cols: number of columns in rectangles
    :param saturation: True or False (optional), default: True
    :return: void
    """

    col_left = floor((rect_cols - 1) / 2)
    col_right = ceil((rect_cols - 1) / 2)
    row_down = ceil((rect_rows - 1) / 2)
    row_up = floor((rect_rows - 1) / 2)

    dirs_names = ['man', 'car', 'nothing', 'wrong_man', 'wrong_car']
    for dir_name in dirs_names:
        files = [file for file in os.listdir(base_path+dir_name) if file.startswith(from_cube)]

        # Load pictures
        imgs, positions, labels = load_RD_detected_imgs(base_path, from_cube, [dir_name], [1])

        positions = positions.reshape(-1, 2).astype(int)

        for i in range(len(files)):
            for source_file in os.listdir(source_dir):
                if not source_file.startswith(to_cube + files[i][6:11]):
                    continue

                range_doppler = load_data(source_dir + source_file, rows*columns)

                # process data
                rd = range_doppler / to_cube_scalar

                rd = np.reshape(rd, (rows, columns))
                rd = np.delete(rd, np.linspace(columns / 2, columns - 1, columns / 2).astype(int), axis=1)  # delete right side
                rd = np.fft.fftshift(rd, axes=0)  # flip image

                # saturation
                if saturation == True:
                    with np.nditer(rd, op_flags=['readwrite']) as it:
                        for elem in it:
                            if elem > 1.0:
                                elem[...] = 1.0

                # Pad data and take rectangle
                row_max = positions[i][0]
                col_max = positions[i][1]
                rd = np.pad(rd, ((rect_rows, rect_rows), (rect_cols, rect_cols)), mode='constant', constant_values=((0, 0), (0, 0)))
                rect = rd[row_max - row_down: row_max + row_up+1, col_max - col_left: col_max + col_right+1]  # array[start:stop+1]

                if saturation == False:
                    rect /= rect.max()

                # Save rectangle if file doesn't exists
                if not os.path.isfile(end_dir + dir_name + '/' + to_cube + files[i][6:len(files)]):
                    f = open(end_dir + dir_name + '/' + to_cube + files[i][6:len(files)], 'w')
                    np.savetxt(f, np.asarray([row_max, col_max]), delimiter=',')
                    np.savetxt(f, rect, delimiter=',')
                    f.close()
                else:
                    print("Error: File already exists")

            print(files[i])


convert_cubes('C:/Users/Ivan/Documents/Geolux/data/Snimka2/', 'C:/Users/Ivan/Documents/Geolux/data/Snimka2_no_saturation/',
              'D:/Geolux/Snimka2/', 'Cube-2', 'Cube-0', 2e4*5, 256, 512, 64, 8, saturation=False)

