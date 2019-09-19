import tkinter as tk

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
        white = 'white ->' + args[1]

    txt_gui.delete(1.0, tk.END)
    txt_gui.insert(tk.CURRENT, red + "   |||   " + orange + "   |||   " + white)


def next_img(root):
    global red, orange, white, exit_flag
    if red == 'NOT' or orange == 'NOT' or white == 'NOT':
        return

    red = 'NOT'
    orange = 'NOT'
    white = 'NOT'
    exit_flag = True
    root.destroy()
    return


def run_gui():
    root = tk.Tk()
    root.geometry("600x400+700+200")

    global exit_flag
    exit_flag = False

    man = tk.Button(master=root, text='man', command=lambda: classify('red:man'), bg='red').place(x=1, y=5, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('red:car'), bg='red').place(x=1, y=40, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('red:nothing'), bg='red').place(x=1, y=75, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('red:wrong man'), bg='red').place(x=120, y=5, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('red:wrong car'), bg='red').place(x=120, y=40, width=100, height=30)

    man = tk.Button(master=root, text='man', command=lambda: classify('orange:man'), bg='orange').place(x=1, y=120, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('orange:car'), bg='orange').place(x=1, y=155, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('orange:nothing'), bg='orange').place(x=1, y=190, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('orange:wrong man'), bg='orange').place(x=120, y=120, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('orange:wrong car'), bg='orange').place(x=120, y=155, width=100, height=30)

    man = tk.Button(master=root, text='man', command=lambda: classify('white:man'), bg='white').place(x=1, y=235, width=100, height=30)
    car = tk.Button(master=root, text='car', command=lambda: classify('white:car'), bg='white').place(x=1, y=270, width=100, height=30)
    nothing = tk.Button(master=root, text='nothing', command=lambda: classify('white:nothing'), bg='white').place(x=1, y=305, width=100, height=30)

    wrong_man = tk.Button(master=root, text='wrong_man', command=lambda: classify('white:wrong man'), bg='white').place(x=120, y=235, width=100, height=30)
    wrong_car = tk.Button(master=root, text='wrong_car', command=lambda: classify('white:wrong car'), bg='white').place(x=120, y=270, width=100, height=30)

    global txt_gui
    txt_gui = tk.Text(master=root, height=1, width=200, bg='#B4B4B4')
    txt_gui.pack(side=tk.BOTTOM)
    txt_gui.insert(tk.CURRENT, red + "   |||   " + orange + "   |||   " + white)

    next = tk.Button(master=root, text = 'NEXT', command=lambda: next_img(root), bg='green').place(x=450, y=135, width=90, height=60)

    while not exit_flag:
        root.update_idletasks()
        root.update()