import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import floor, ceil
from os import listdir
from csv import reader
from time import time
import label_gui


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


def overlap(row1, col1, row2, col2, rect_rows, rect_cols):
    """
    Function is calculating overlap of two rectangles in percentage (0 - no overlap, 1 - same rectangle)

    :param row1: center row of first rectangle
    :param col1: center column of first rectangle
    :param row2: center row of second rectangle
    :param col2: center column of second rectangle
    :param rect_rows: number of rows in rectangle
    :param rect_cols: number of columns of rectangle
    :return: overlap percentage
    """

    col_left = floor( (rect_cols-1)/2 )
    col_right = ceil( (rect_cols-1)/2 )
    row_down = ceil( (rect_rows-1)/2 )
    row_up = floor( (rect_rows-1)/2 )

    if row1 < row2:
        row1, row2 = row2, row1
        col1, col2 = col2, col1

    if col1 > col2:
        x = max(0, (col2+col_right) - (col1-col_left) + 1)
    else:
        x = max(0, (col1+col_right) - (col2-col_left) + 1)
    y = max(0, (row2+row_up) - (row1-row_down) + 1)

    return (x*y) / (rect_cols*rect_rows)


def find_rects(image, max_indices, rows, columns, rect_rows, rect_columns, max_overlap):
    """
    Function is using indices of maximum amplitudes (in array) to calculate row and column index of maximum amplitude.
    Also function is performing padding of whole range doppler graph, so rectangles are allowed to go be partly outside
    range doppler graph. Padded part and colored rectangles are added to the image and ready for display.
    Row and column indices of maximums (3 maximums) are adjusted so they include the padding

    :param image: Colored greyscale image as numpy ndarray (3D matrix)
    :param max_indices: numpy array of indices of sorted amplitude maximums
    :param rows: number of rows in range doppler graph
    :param columns: number of columns in range doppler graph
    :param rect_rows: number of rows in rectangle
    :param rect_columns: number of columns in rectangle
    :param max_overlap: maximum overlap of two rectangles in percentage
    :return: image - padded image of range doppler graph with border colored rectangles (3 rectangles)
             row_max - numpy array of indices of center rows of 3 rectangles in image
             col_max - numpy array of indices of center columns of 3 rectangles in image
    """

    image = np.pad(image, ((rect_rows, rect_rows), (rect_columns, rect_columns), (0, 0)),
                   mode='constant', constant_values=((0,0), (0,0), (0,0)))

    row_max = np.zeros((3,)).astype(int);
    col_max = np.zeros((3,)).astype(int)

    col_left = floor((rect_columns - 1) / 2)
    col_right = ceil((rect_columns - 1) / 2)
    row_down = ceil((rect_rows - 1) / 2)
    row_up = floor((rect_rows - 1) / 2)

    i = 0
    for cnt in range(3):
        cond_met = False; rect_cant_be_found = False
        while not cond_met and not rect_cant_be_found:
            row_max[cnt] = floor( max_indices[i] / columns )
            col_max[cnt] = max_indices[i] % columns

            # adjust indices for padded elements
            row_max[cnt] += rect_rows
            col_max[cnt] += rect_columns

            # check the overlap with each saved rectangle
            for j in range(cnt):
                if overlap(row_max[j], col_max[j], row_max[cnt], col_max[cnt], rect_rows, rect_columns) < max_overlap:
                    cond_met = True
                else:
                    cond_met = False
                    break
            if cnt == 0:
                cond_met = True

            i += 1
            if i >= max_indices.size:
                rect_cant_be_found = True
                print("cant be found")

        # rectangle - amplitude at center is at index (32, 3)
        rect = image[row_max[cnt]-row_down : row_max[cnt]+row_up+1, col_max[cnt]-col_left : col_max[cnt]+col_right+1]  # array[start:stop+1]

        # draw rectangle
        # fig = plt.figure('Rectangle')
        # plt.cla()
        # plt.imshow(rect)
        # plt.pause(0.00001)

        # color rectangle border
        if cnt == 0:
            color = np.array([1.0, 0.0, 0.0])  # red
        elif cnt == 1:
            color = np.array([1.0, 0.647, 0.0])  # orange
        else:
            color = np.array([1.0, 1.0, 1.0])  # white

        image[row_max[cnt]-row_down : row_max[cnt]+row_up+1, col_max[cnt]-col_left] = color  # left line
        image[row_max[cnt]-row_down : row_max[cnt]+row_up+1, col_max[cnt]+col_right] = color  # right line
        image[row_max[cnt]-row_down, col_max[cnt]-col_left : col_max[cnt]+col_right+1] = color  # bottom line
        image[row_max[cnt]+row_up, col_max[cnt]-col_left : col_max[cnt]+col_right+1] = color  # upper line

    return image, row_max, col_max


def draw (dir_path, cube, scalar, rows, columns, rect_rows, rect_columns, max_overlap):
    """
    Function is displaying image of every file in given directory. It scales the data using scalar parameter,
    converts data to image (colors the data using 'plasma' colormap) and pads the image.
    It also finds and draws rectangles of given shape and maximum overlap condition.

    :param dir_path: path to directory where all the data is stored (each image in its own file)
    :param cube: cube to be used, only files of given cube will be processed
    :param scalar: number which scales all the data in every image
    :param rows: number of rows in range doppler graph
    :param columns: number of columns in range doppler graph
    :param rect_rows: number of rows in rectangles
    :param rect_columns: number of columns in rectangles
    :param max_overlap: maximum overlap allowed between any two rectangles
    """

    for file in listdir(dir_path):
        start = time()
        if not file.startswith(cube):
            continue

        file_path = dir_path + file

        # range_doppler2 = np.genfromtxt(file_path, delimiter=',')
        range_doppler = load_data(file_path, rows*columns)

        rd = range_doppler / scalar
        # print(rd.max())

        cmap = cm.get_cmap('plasma')

        # process data
        rd = np.reshape(rd, (rows, columns))
        rd = np.delete(rd, np.linspace(columns/2, columns-1, columns/2).astype(int), axis=1)  # discard the right side (blank)
        rd = np.fft.fftshift(rd, axes=0)  # flip image

        # find indices of maximums
        rd = np.reshape(rd, (rd.size,))
        max_indices = np.flip(np.argsort(rd))

        # saturation
        for i in range(rd.size):
            if rd[i] > 1.0:
                rd[i] = 1.0

        # add color to image
        rd = np.reshape(rd, (rows, -1))
        color = cmap(rd)  # from 0.0 to 1.0
        img = np.delete(color, 3, axis=2)  # rgba - discarding a

        # find rectangles
        img, row_max, col_max = find_rects(img, max_indices, img.shape[0], img.shape[1], rect_rows, rect_columns, max_overlap)

        plt.figure('Image')
        plt.cla()
        plt.imshow(img, aspect='auto')
        plt.title(file)
        plt.pause(000000.1)

        # label_gui.run_gui('C:/Users/Ivan/Documents/Geolux/data/Snimka2/', file, rd, row_max, col_max, rect_rows, rect_columns)

        stop = time()
        # print(stop-start)
        print(file)
    plt.show()


def find_max_avg_in_all(dir_path):
    """
    This is a helper function to get some insight in data. Since it's a standalone function all parameters can
    be changed from within the function as it suites you best.
    Function is calculating maximum over all the data, average and number of data values bigger than some amount
    (in percentage) for each file in given directory.

    :param dir_path: directory in which all files are stored
    :return: void
    """
    max = 0
    for file in listdir(dir_path):
        if not file.startswith('Cube-0'):
            continue

        file_path = dir_path + file
        f = open(file_path, 'r')
        cs = reader(f)
        cnt = 0; br = 0; suma = 0
        for row in cs:
            for column in row:
                if float(column) > max:
                    max = float(column)
                if float(column) > 18000/2:
                    cnt += 1
                suma+=float(column)
                br += 1

        f.close()
        print(file)
        print('cnt (>9000)[%] = {}'.format(cnt/br))
        print('max = {}'.format(max))
        print('avg = {}'.format(float(suma)/float(br)))
    print(max)


def remove_first_n_characters(dir_path, n):
    """
    Function removes first n characters from all the files in given directory

    :param dir_path: path to the directory
    :param n: number of characters to be removed from files
    :return: void
    """
    for file in listdir(dir_path):
        if file[0] == '.':
            continue

        file_path = dir_path + file
        f = open(file_path, 'r+')
        rd = f.read()
        f.close()

        f = open(file_path, 'w')
        f.write(rd[n+1 : len(rd)-1])
        f.close()

        print(file)


# izbaci_prva_dva_broja('D:/Geolux/Snimka2/', 7)
# find_max_avg_in_all('D:\Geolux\Snimka1/')
draw('D:/Geolux/Snimka1/', 'Cube-2', 2e3*5, 256, 512, 64, 8, 0.3)