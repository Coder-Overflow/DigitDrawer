import numpy as np
from tkinter import *
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

root = Tk()
root.title("DigitDrawer")

CANVAS_SIZE = (250, 200)
canvas = np.zeros(CANVAS_SIZE[::-1])

model = tf.keras.models.load_model('model.h5')

def paint(event):
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "black"
    # display the mouse movement inside canvas

    #print(f"{event.x-5} - {event.x + 5}", f"{event.y-5} - {event.y + 5}")

    for y in range(-5, 6):
        for x in range(-5, 6):
            try:
                canvas[event.y+y, event.x+x] = 1
            except:
                pass

    wn.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    image = Image.fromarray(canvas)
    image = image.resize((28, 28))
    imageArray = np.array(image)

    global model
    prediction = model(np.reshape(imageArray, (1, 28, 28))).numpy()

    predictionNum = prediction.argmax(axis=1)[0]
    predictionProb = prediction.max(axis=1)[0]

    global label
    label['text'] = f"{predictionNum} with a probability of {round(predictionProb * 100, 2)}%"

def clearDrawing():
    wn.delete("all")
    
    global canvas
    canvas = np.zeros(CANVAS_SIZE[::-1])

    global label
    label["text"] = "Start drawing..."

def showImg():
    image = Image.fromarray(canvas)
    image = image.resize((28, 28), Image.ANTIALIAS)
    imageArray = np.array(image)

    print(imageArray.shape)

    plt.imshow(imageArray, cmap='gray')
    plt.show()

label = Label(text="Start drawing...")
label.pack()

wn = Canvas(root, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], bg='white')
wn.bind('<B1-Motion>', paint)
wn.pack()

button = Button(text="Clear", command=clearDrawing)
button.pack()

showImg = Button(text="Show Image", command=showImg)
showImg.pack()

root.mainloop()