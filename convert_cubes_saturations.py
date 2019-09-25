import numpy as np
import os
from csv import reader
from load_RD_detected_imgs import load_RD_detected_imgs
from draw_saved_rectangle import draw_saved_rectangle


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


def convert_cubes(base_path, end_dir, source_dir, from_cube, to_cube, to_cube_scalar, rows, columns, saturation=True):
    dirs_names = ['man', 'car', 'nothing', 'wrong_man', 'wrong_car']
    for dir_name in dirs_names:
        files = os.listdir(base_path+dir_name)
        # Load pictures
        imgs, positions, labels = load_RD_detected_imgs(base_path, from_cube, [dir_name], [1])

        positions = positions.reshape(-1, 2).astype(int)

        for i in range(len(files)):
            for source_file in os.listdir(source_dir):
                if not source_file.startswith(to_cube + files[i][6:11]):
                    continue

                range_doppler = load_data(source_dir + source_file)
                rd = range_doppler / to_cube_scalar

                # process data
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
                rd = np.pad(rd, ((64, 64), (8, 8)), mode='constant', constant_values=((0, 0), (0, 0)))
                rect = rd[row_max - 32: row_max + 32, col_max - 3: col_max + 5]  # array[start:stop-1]

                # Save rectangle if file doesn't exists
                if not os.path.isfile(end_dir + dir_name + '/' + to_cube + files[i][6:len(files)]):
                    f = open(end_dir + dir_name + '/' + to_cube + files[i][6:len(files)], 'w')
                    np.savetxt(f, np.asarray([row_max, col_max]), delimiter=',')
                    np.savetxt(f, rect, delimiter=',')
                    f.close()

                print(files[i])


# convert_cubes('C:/Users/Ivan/Documents/Geolux/data/Snimka1/', 'C:/Users/Ivan/Documents/Geolux/data/Snimka1/',
#               'D:/Geolux/Snimka1/', 'Cube-2', 'Cube-0', 2e4*5, 256, 512)


def rename_files(dir_path):
    for file in os.listdir(dir_path):
        src = dir_path + file
        if file.startswith('red_'):
            dest = dir_path + file[4:len(file)-4] + '_red' + '.txt'
        elif file.startswith('orange_'):
            dest = dir_path + file[7:len(file)-4] + '_orange' + '.txt'
        elif file.startswith('white_'):
            dest = dir_path + file[6:len(file)-4] + '_white' + '.txt'
        else:
            print("Error: file - {}". format(file))
            return

        os.rename(src, dest)
