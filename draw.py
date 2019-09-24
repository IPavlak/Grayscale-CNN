import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import floor, ceil
from os import listdir
from csv import reader
from time import time
import label_gui

def load_data(file_path):
    f = open(file_path, 'r')
    nump = np.zeros((512 * 256))
    cs = reader(f)
    cnt = 0
    for row in cs:
        for col in row:
            nump[cnt] = float(col)
            cnt += 1

    f.close()
    return nump


def overlap(row1, col1, row2, col2, rect_rows, rect_cols):
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


def find_rects(image, max_indices, rows, columns, rect_rows, rect_columns):
    image = np.pad(image, ((rect_rows, rect_rows), (rect_columns, rect_columns), (0, 0)),
                   mode='constant', constant_values=((0,0), (0,0), (0,0)))

    row_max = np.zeros((3,)).astype(int); col_max = np.zeros((3,)).astype(int)
    i = 0
    for cnt in range(3):
        cond_met = False; rect_cant_be_found = False
        while not cond_met and not rect_cant_be_found:
            row_max[cnt] = floor( max_indices[i] / columns )
            col_max[cnt] = max_indices[i] % columns

            # adjust indices for padded elements
            row_max[cnt] += rect_rows
            col_max[cnt] += rect_columns

            for j in range(cnt):
                if overlap(row_max[j], col_max[j], row_max[cnt], col_max[cnt], rect_rows, rect_columns) < 0.3:
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

        # rectangle - amplitude at center is at indices (32, 3)
        rect = image[row_max[cnt]-32 : row_max[cnt]+32, col_max[cnt]-3 : col_max[cnt]+5]  # array[start:stop-1]

        # draw rectangle
        # fig = plt.figure('Rectangle')
        # plt.cla()
        # plt.imshow(rect)
        # plt.pause(0.00001)

        # color rectangle
        if cnt == 0:      # 128 < row_max[cnt] < 270 and col_max[cnt] < 128:
            color = np.array([1.0, 0.0, 0.0])  # red
        elif cnt == 1:              # 128 <= col_max[cnt] <= 150:
            color = np.array([1.0, 0.647, 0.0])  # orange
        else:
            color = np.array([1.0, 1.0, 1.0])  # white

        image[row_max[cnt]-32 : row_max[cnt]+32, col_max[cnt]-3] = color  # left line
        image[row_max[cnt]-32 : row_max[cnt]+32, col_max[cnt]+4] = color  # right line
        image[row_max[cnt]-32, col_max[cnt]-3 : col_max[cnt]+5] = color  # bottom line
        image[row_max[cnt]+31, col_max[cnt]-3 : col_max[cnt]+5] = color  # upper line

    return image, row_max, col_max


def draw (dir_path, scalar, rows, columns):
    for file in listdir(dir_path):
        start = time()
        if not file.startswith('Cube-2'):
            continue

        file_path = dir_path + file

        # range_doppler2 = np.genfromtxt(file_path, delimiter=',')
        range_doppler = load_data(file_path)

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

        # add color to image
        for i in range(rd.size):
            if rd[i] > 1.0:
                rd[i] = 1.0

        rd = np.reshape(rd, (rows, -1))
        color = cmap(rd)  # from 0.0 to 1.0
        img = np.delete(color, 3, axis=2)  # rgba - discarding a

        # finding rectangles
        img, row_max, col_max = find_rects(img, max_indices, img.shape[0], img.shape[1], 64, 8)

        plt.figure('Image')
        plt.cla()
        plt.imshow(img, aspect='auto')
        plt.title(file)
        plt.pause(000000.1)

        # label_gui.run_gui('C:/Users/Ivan/Documents/Geolux/data/Snimka2/', file, rd, row_max, col_max)

        stop = time()
        print(file)
    plt.show()

def find_max_avg_in_all(dir_path):
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
        print('avg = {}'.format(float(suma)/float(br)))
    print(max)


def izbaci_prva_dva_broja(dir_path):
    for file in listdir(dir_path):
        if file[0] == '.':
            continue

        file_path = dir_path + file
        f = open(file_path, 'r+')
        rd = f.read()
        f.close()

        f = open(file_path, 'w')
        f.write(rd[8:len(rd)-1])
        f.close()

        print(file)

# izbaci_prva_dva_broja('D:/Geolux/Snimka2/')
# find_max_avg_in_all('D:\Geolux\Snimka1/')
draw('D:/Geolux/Snimka1/', 2e3*5, 256, 512)