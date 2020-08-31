#To Run
#type "python Testing-LeNet-Model-using-Tkinter.py"


#Import Library
import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *


##Loading Model and performing predictions
model=tf.keras.models.load_model('Digit-mnist-LeNet.h5')
print(model.summary())

#Takes image and process the image and perform prediction
def perform_prediction():
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img=img/255.0
    pred=model.predict(img)
    return pred

#Calls the model and take predictions and display the predictions
def model_pred():
    filename = "image.png"
    image1.save(filename)
    pred = perform_prediction()
    class_name = classes[np.argmax(pred[0])]
    class_score = pred[0][np.argmax(pred[0])]
    txt.insert(tk.INSERT,"{}\nAccuracy: {}%".format(class_name,int(class_score*100)))


classes = [0,1,2,3,4,5,6,7,8,9]
width = 200
height = 200
center = height//2
white =(255,255,255)
green = (0,128,0)

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1,y1,x2,y2, fill = "black",width = 8)
    draw.line([x1,y1,x2,y2],fill = "black", width =8)
    
def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 200, 200), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


root = Tk()
root.geometry('200x310') 
root.resizable(0,0)
cv = Canvas(root, width=width, height=height, bg= 'midnight blue')
cv.pack()
# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt=tk.Text(root,bd=3,exportselection=0,bg='midnight blue',fg="white",font='Helvetica',padx=10,pady=2,height=5,width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
##button=Button(text="save",command=save)
btnModel=Button(text="Guess it!",command=model_pred,bg = 'gold')
btnClear=Button(text="Clear",command=clear)
##button.pack()
btnModel.pack()
btnClear.pack()
txt.pack()
root.title('Digit Recognition')
root.mainloop()