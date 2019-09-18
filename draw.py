import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import floor, ceil
from os import listdir
from csv import reader
from time import time

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


def find_rects(image, max_indices, rows, columns):
    image = np.pad(image, ((64, 64), (8, 8), (0, 0)), mode='constant', constant_values=((0,0), (0,0), (0,0)))

    i = 0; cond_met = False; rect_cant_be_found = False
    while not cond_met and not rect_cant_be_found:
        row_max = floor( max_indices[i] / columns )
        col_max = max_indices[i] % columns
        cond_met = True
        i += 1
        if i >= max_indices.size:
            rect_cant_be_found = True

    # adjust indices for padded elements
    row_max += 64
    col_max += 8

    # amplitude at center is at indices (32, 3)
    rect = image[row_max-32 : row_max+32, col_max-3 : col_max+5]  # array[start:stop-1]

    # color rectangle red
    image[row_max-32 : row_max+32, col_max-3] = np.array([1.0, 0.0, 0.0])  # left line
    image[row_max-32 : row_max+32, col_max+4] = np.array([1.0, 0.0, 0.0])  # right line
    image[row_max-32, col_max-3 : col_max+5] = np.array([1.0, 0.0, 0.0])  # bottom line
    image[row_max+31, col_max-3 : col_max+5] = np.array([1.0, 0.0, 0.0])  # upper line

    return image


def draw (dir_path, scaler, rows, columns):
    for file in listdir(dir_path):
        start = time()
        if not file.startswith('Cube-0'):
            continue

        file_path = dir_path + '\\' + file

        # range_doppler2 = np.genfromtxt(file_path, delimiter=',')
        range_doppler = load_data(file_path)

        rd = range_doppler / scaler
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
        img = find_rects(img, max_indices, img.shape[0], img.shape[1])

        plt.cla()
        plt.imshow(img, aspect='auto')
        plt.title(file)
        plt.pause(000000.1)
        stop = time()
        # print(stop-start)
    plt.show()

def find_max_avg_in_all(dir_path):
    max = 0
    for file in listdir(dir_path):
        if not file.startswith('Cube-0'):
            continue

        file_path = dir_path + '\\' + file
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

        file_path = dir_path + '\\' + file
        f = open(file_path, 'r+')
        rd = f.read()
        f.close()

        f = open(file_path, 'w')
        f.write(rd[8:len(rd)-1])
        f.close()

        print(file)

# izbaci_prva_dva_broja('D:\Geolux\Snimka1')
# find_max_avg_in_all('D:\Geolux\Snimka1')
draw('D:\Geolux\Snimka1', 2e4*5, 256, 512)