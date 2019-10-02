import tkinter as tk
import numpy as np
from math import floor, ceil

red = 'NOT'
orange = 'NOT'
white = 'NOT'
txt_gui = None
exit_flag = False


def classify(ids):
    global red, orange, white
    args = ids.split(':')

    if args[0] == 'red':
        red = 'red -> ' + args[1]
    elif args[0] == 'orange':
        orange = 'orange -> ' + args[1]
    elif args[0] == 'white':
        white = 'white -> ' + args[1]

    txt_gui.delete(1.0, tk.END)
    txt_gui.insert(tk.CURRENT, red + "   |||   " + orange + "   |||   " + white)


def next_img(root):
    global red, orange, white, exit_flag

    red = 'NOT'
    orange = 'NOT'
    white = 'NOT'
    exit_flag = True
    root.destroy()
    return


def save_img(root, base_path, file, rd, row_max, col_max, rect_rows, rect_cols):
    global red, orange, white, exit_flag
    if red == 'NOT' or orange == 'NOT' or white == 'NOT':
        return

    rd = np.pad(rd, ((rect_rows, rect_rows), (rect_cols, rect_cols)), mode='constant', constant_values=((0,0), (0,0)))

    col_left = floor((rect_cols - 1) / 2)
    col_right = ceil((rect_cols - 1) / 2)
    row_down = ceil((rect_rows - 1) / 2)
    row_up = floor((rect_rows - 1) / 2)

    rect_red = rd[row_max[0]-row_down : row_max[0]+row_up+1, col_max[0]-col_left : col_max[0]+col_right+1]  # array[start:stop+1]
    rect_orange = rd[row_max[1]-row_down : row_max[1]+row_up+1, col_max[1]-col_left : col_max[1]+col_right+1]  # array[start:stop+1]
    rect_white = rd[row_max[2]-row_down : row_max[2]+row_up+1, col_max[2]-col_left : col_max[2]+col_right+1]  # array[start:stop+1]

    rect_red = np.reshape(rect_red, (rect_rows, rect_cols))
    rect_orange = np.reshape(rect_orange, (rect_rows, rect_cols))
    rect_white = np.reshape(rect_white, (rect_rows, rect_cols))


    f = open(base_path + red[7:len(red)] + '/' + file[0:len(file)-4] + '_red.txt', 'w')
    rect_red_array = np.asarray(rect_red)
    np.savetxt(f, np.asarray([row_max[0], col_max[0]]), delimiter=',')
    np.savetxt(f, rect_red_array, delimiter=',')
    f.close()

    f = open(base_path + orange[10:len(orange)] + '/' + file[0:len(file)-4] + '_orange.txt', 'w')
    rect_orange_array = np.asarray(rect_orange)
    np.savetxt(f, np.asarray([row_max[1], col_max[1]]), delimiter=',')
    np.savetxt(f, rect_orange_array, delimiter=',')
    f.close()

    f = open(base_path + white[9:len(white)] + '/' + file[0:len(file)-4] + '_white.txt', 'w')
    rect_white_array = np.asarray(rect_white)
    np.savetxt(f, np.asarray([row_max[2], col_max[2]]), delimiter=',')
    np.savetxt(f, rect_white_array, delimiter=',')
    f.close()

    red = 'NOT'
    orange = 'NOT'
    white = 'NOT'
    exit_flag = True
    root.destroy()
    return


def run_gui(base_path, file, rd, row_max, col_max, rect_rows, rect_cols):
    root = tk.Tk()
    root.geometry("600x400+700+200")  # size and placement of the gui window respectively

    global exit_flag
    exit_flag = False

    man = tk.Button(master=root, text='man', command=lambda: classify('red:man'), bg='red').place(x=1, y=5, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('red:car'), bg='red').place(x=1, y=40, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('red:nothing'), bg='red').place(x=1, y=75, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('red:wrong_man'), bg='red').place(x=120, y=5, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('red:wrong_car'), bg='red').place(x=120, y=40, width=100, height=30)
    garbage = tk.Button(master=root, text='garbage', command=lambda: classify('red:garbage'), bg='red').place(x=120, y=75, width=100, height=30)

    man = tk.Button(master=root, text='man', command=lambda: classify('orange:man'), bg='orange').place(x=1, y=120, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('orange:car'), bg='orange').place(x=1, y=155, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('orange:nothing'), bg='orange').place(x=1, y=190, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('orange:wrong_man'), bg='orange').place(x=120, y=120, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('orange:wrong_car'), bg='orange').place(x=120, y=155, width=100, height=30)
    garbage = tk.Button(master=root, text='garbage', command=lambda: classify('orange:garbage'), bg='orange').place(x=120, y=190, width=100, height=30)

    man = tk.Button(master=root, text='man', command=lambda: classify('white:man'), bg='white').place(x=1, y=235, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('white:car'), bg='white').place(x=1, y=270, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('white:nothing'), bg='white').place(x=1, y=305, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('white:wrong_man'), bg='white').place(x=120, y=235, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('white:wrong_car'), bg='white').place(x=120, y=270, width=100, height=30)
    garbage = tk.Button(master=root, text='garbage', command=lambda: classify('white:garbage'), bg='white').place(x=120, y=305, width=100, height=30)

    global txt_gui
    txt_gui = tk.Text(master=root, height=1, width=200, bg='#B4B4B4')
    txt_gui.pack(side=tk.BOTTOM)
    txt_gui.insert(tk.CURRENT, red + "   |||   " + orange + "   |||   " + white)

    next = tk.Button(master=root, text = 'NEXT', command=lambda: save_img(root, base_path, file, rd, row_max, col_max, rect_rows, rect_cols), bg='green').place(x=450, y=135, width=90, height=60)
    skip = tk.Button(master=root, text = 'SKIP', command=lambda: next_img(root), bg='grey').place(x=450, y=245, width=90, height=60)

    while not exit_flag:
        root.update_idletasks()
        root.update()