import tkinter as tk
import numpy as np
import sys
import tensorflow as tf
import time
import cv2

size = (150, 150)
model = tf.keras.models.load_model(sys.argv[1])
img = cv2.imread(sys.argv[2])
img = cv2.resize(img, size)
classification = model.predict(
            [np.array(img).reshape(1, 150, 150, 3)]
)
print(classification.argmax())

window = tk.Tk()

label = tk.Label(window, text="Coming soon...")
label.pack()

window.mainloop()