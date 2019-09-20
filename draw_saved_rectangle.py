import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def draw_saved_rectangle(file):
    cmap = cm.get_cmap('plasma')
    with open(file, 'r') as f:
        row_max = int(float(f.readline().rstrip()))
        col_max = int(float(f.readline().rstrip()))
        rd = np.loadtxt(f, delimiter=',')

        color = cmap(rd)  # from 0.0 to 1.0
        img = np.delete(color, 3, axis=2)  # rgba - discarding a
        plt.imshow(img)
        plt.show()


draw_saved_rectangle('C:\\Users\\Ivan\\Documents\\Geolux\\data\\garbage\\red_Cube-2-0005.txt')