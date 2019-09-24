import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def draw_saved_rectangle(file = None, rd = None):
    if file is None and rd is None:
        print("Error: draw_saved_rectangle called with both None parameters")
        return

    cmap = cm.get_cmap('plasma')
    if rd is None:
        with open(file, 'r') as f:
            row_max = int(float(f.readline().rstrip()))
            col_max = int(float(f.readline().rstrip()))
            rd = np.loadtxt(f, delimiter=',')

    with np.nditer(rd, op_flags=['readwrite']) as it:
        for elem in it:
            if elem > 1.0:
                elem[...] = 1.0

    color = cmap(rd)  # from 0.0 to 1.0
    img = np.delete(color, 3, axis=2)  # rgba - discarding a
    plt.imshow(img)
    plt.show()



# draw_saved_rectangle('C:/Users/Ivan/Documents/Geolux/data/garbage/red_Cube-2-0005.txt')