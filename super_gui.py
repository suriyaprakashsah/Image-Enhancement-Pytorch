from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends import backend_tkagg
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import tkinter as tk
from tkinter import ttk
from skimage import io
import cv2
from super_resolve import resolve
from PIL import Image, ImageTk

style.use('ggplot')
warnings.simplefilter('ignore')

from tkinter import filedialog


global_filename = ""

def get_input(inp):
    print(inp)


def browsefunc():
    global global_filename
    filename = filedialog.askopenfilename()
    global_filename = filename
    pathlabel.config(text=filename)


def get_img_name(path):
    path_split = path.split("/")
    return path_split[-1]


def save_file(image, img_path):
    img_name = get_img_name(img_path)
    save_img_name = img_name[:-4] + "_SR_" + img_name[-4:]

    save_folder =  filedialog.askdirectory()
    save_file = save_folder + "/" + save_img_name

    io.imsave(save_file, image)


def show_lr(path):
    popup_lr = tk.Tk()
    popup_lr.wm_title("Image Processing Using CNN")

    label = ttk.Label(popup_lr, justify=tk.LEFT, text="""Original Image""", font=("Verdana", 14, "bold"))
    label.pack(side="top", fill="x", pady=30, padx=30)

    img = io.imread(path)
    if img is None:
        print(path)
        print(type(path))
        print("IMG IS NONE")
        
    plt.imshow(img)
    fig, ax = plt.subplots()
    im = ax.imshow(img, origin='upper')
    plt.grid("off")

    canvas = backend_tkagg.FigureCanvasTkAgg(fig, popup_lr)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = backend_tkagg.NavigationToolbar2Tk(canvas, popup_lr)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B1 = ttk.Button(popup_lr, text="SELECT FOLDER TO SAVE THIS IMAGE", command=lambda: save_file(img, path, scale=1))
    B1.pack(side="top")

    label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B2 = ttk.Button(popup_lr, text="CLOSE THIS WINDOW", command=popup_lr.destroy)
    B2.pack(side="top")

    popup_lr.mainloop()


def show_sr(path, model):
    popup_sr = tk.Tk()
    popup_sr.wm_title("Image Processing Using CNN")

    label = ttk.Label(popup_sr, justify=tk.CENTER, text="""Super Resoved Image Using """ + model , font=("Verdana", 14, "bold"))
    label.pack(side="top", fill="x", pady=10, padx=30)
    
    resolve(input_path=path, model=model+'_model_path.pth')

    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    plt.imshow(img)
    fig, ax = plt.subplots()
    im = ax.imshow(img, origin='upper')
    plt.grid("off")

    canvas = backend_tkagg.FigureCanvasTkAgg(fig, popup_sr)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = backend_tkagg.NavigationToolbar2Tk(canvas, popup_sr)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    label = ttk.Label(popup_sr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B1 = ttk.Button(popup_sr, text="SELECT FOLDER TO SAVE THIS IMAGE", command=lambda: save_file(img, path))
    B1.pack(side="top")

    label = ttk.Label(popup_sr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)
    
    B2 = ttk.Button(popup_sr, text="CLOSE THIS WINDOW", command = popup_sr.destroy)
    B2.pack(side= "top")

    popup_sr.mainloop()



root = tk.Tk()
tk.Tk.wm_title(root, "Super Resolution GUI")
label = ttk.Label(root, text="Super Resolution Using CNN", font=("Verdana", 22, "bold"))
label.pack(side="top", pady=30, padx=50)

desc = '''Super Resolution Model with diverse CNN features and manage interaction with the systems.
All you need to do is to drag and drop, and the rest would be managed by the GUI.'''
label = ttk.Label(root, justify=tk.CENTER, text=desc, font=("Verdana", 11))
label.pack(side="top", pady=30, padx=30)

label = ttk.Label(root, justify=tk.CENTER,
                  text="Click the browse button below to select the image file", font=("Verdana", 11))
label.pack(side="top", pady=5, padx=30)


button1 = ttk.Button(root, text="BROWSE", command=lambda: browsefunc())
button1.pack()

label = ttk.Label(root, justify=tk.CENTER, text="Path of the selected image file", font=("Verdana", 11))
label.pack(side="top", pady=3, padx=30)

pathlabel = ttk.Label(root, font=("Verdana", 11, "bold"))
pathlabel.pack(side="top", pady=3, padx=30)

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=1, padx=30)

button1 = ttk.Button(root, text="SHOW ORIGINAL IMAGE", command=lambda: show_lr(global_filename))
button1.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button2 = ttk.Button(root, text="Super Resolution using Convolution Neural Network", command=lambda: show_sr(global_filename, model='SRCNN'))
button2.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button3 = ttk.Button(root, text="Fast Super Resolution using Convolutional Neural Network", command=lambda: show_sr(global_filename, model='FSRCNN'))
button3.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button4 = ttk.Button(root, text="SubPixel Convolutional Neural Network", command=lambda: show_sr(global_filename, model='SUBPIX'))
button4.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)


button5 = ttk.Button(root, text="Super-Resolution Using Very Deep Convolutional Network", command=lambda: show_sr(global_filename, model='VDSR'))
button5.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)

button6 = ttk.Button(root, text="Enhanced Deep Residual Networks for Single Image Super-Resolution", command=lambda: show_sr(global_filename, model='EDSR'))
button6.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)

button7 = ttk.Button(root, text="Super-Resolution using Generative Adversarial Network", command=lambda: show_sr(global_filename, model='SRGAN'))
button7.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)


button8 = ttk.Button(root, text="QUIT", command=lambda: show_sr(exit(0)))
button8.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)

if __name__ == "__main__":
    root.mainloop()
