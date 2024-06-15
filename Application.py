from tkinter import filedialog
from tkinter import *
import tkinter as tk
import numpy
import tensorflow as tf
from PIL import ImageTk, Image
from keras.models import load_model
from keras.preprocessing import image as keras_image  # Rename the imported module
import numpy as np

# Load the trained model to classify traffic signs
model = load_model('Newmodel.h5')

# Dictionary to label all traffic signs class.
classes = {0: 'Dyed-lifted-polyps',
           1: 'Dyed-resection-margins',
           2: 'Esophagitis',
           3: 'Normal-Cecum',
           4: 'Normal-Pylorus',
           5: 'Normal-z-line',
           6: 'Polyps',
           7: 'Ulcerative-colitis'
           }

# Initialize GUI
top = tk.Tk()
top.attributes('-fullscreen', True)
top.geometry('800x600')

# Logo image
path = "logo.jpg"
load = Image.open(path)
render = ImageTk.PhotoImage(load)
top.iconphoto(False, render)
frame1 = Frame(top, width=200, height=200)
frame1.pack()
frame1.place(x=0, y=0)
img1 = ImageTk.PhotoImage(Image.open("logo.jpg"))
label1 = Label(frame1, image=img1)
label1.pack()
frame2 = Frame(top, width=200, height=200)
frame2.pack()
frame2.place(x=1160, y=0)
img2 = ImageTk.PhotoImage(Image.open("logo.jpg"))
label2 = Label(frame2, image=img2)
label2.pack()

# Classes images apple and corn
frame3 = Frame(top, width=280, height=221)
frame3.pack()
frame3.place(x=50, y=320)
img3 = ImageTk.PhotoImage(Image.open("1.jpg"))
label3 = Label(frame3, image=img3)
label3.pack()
frame4 = Frame(top, width=280, height=221)
frame4.pack()
frame4.place(x=1040, y=320)
img4 = ImageTk.PhotoImage(Image.open("2.jpg"))
label4 = Label(frame4, image=img4)
label4.pack()

# Title of project
top.title('Medical Image Classification Using Deep Learning Technique')
top.configure(background='#33FFFC')
label = Label(top, background='#FFFFFF', font=('arial', 15, 'bold'))
sign_image = Label(top)


# Classification button Function
def classify(file_path):
    global label_packed
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Corrected the module name to keras_image
    img = np.expand_dims(img, axis=0)  # Reshape the image array
    pred = model.predict(img)[0]
    sign = classes[max(range(len(pred)), key=lambda x: pred[x])]
    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.60, rely=0.885)


# upload image function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(e)


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Medical Image Classification Using Deep Learning Technique", pady=20,
                font=('arial', 16, 'bold'))
heading.configure(background='#FFFFFF', foreground='#364156')
heading.pack()

# Exit button function
exit_button = Button(top, text="      Exit      ", command=top.destroy, padx=10, pady=5)
exit_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
exit_button.place(relx=0.33, rely=0.885)

top.mainloop()
