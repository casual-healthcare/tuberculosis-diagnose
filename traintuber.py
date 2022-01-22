import cv2
import numpy as np
import os
import sys
import tensorflow as tf

# import needed libs

EPOCHS = 10  # number of times to run the training program on every img
IMG_WIDTH = 150
IMG_HEIGHT = 150
NUM_CATEGORIES = 2


def main():
    if len(sys.argv) not in [3, 4]:
        x = "train"
        y = "test"
    else:
        x = sys.argv[1]
        y = sys.argv[2]
    print("Loading training data...")
    ximg, xlab = load_data(x)  # load training images
    print("Successfuly loaded training data!")
    print("Loading testing data...")
    yimg, ylab = load_data(y)  # load testing images
    print("Done loading data!")
    print("Preparing to train neural network...")
    xlab = tf.keras.utils.to_categorical(xlab)
    ylab = tf.keras.utils.to_categorical(ylab)
    xtrain, xlabel = np.array(ximg), np.array(xlab)
    ytest, ylabel = np.array(yimg), np.array(ylab)
    model = get_model()  
    model.fit(xtrain, xlabel, epochs=EPOCHS) # get the model to train
    model.evaluate(ytest,  ylabel, verbose=2)  # test the model to check how well it did
    if len(sys.argv) == 4:  # save function
        filename = sys.argv[3]
        confirm = input("Save? ")
        if confirm.lower() != "n" and confirm.lower() != "no":
            model.save(filename)
            print(f"Model saved to {filename}.")


def load_data(data_dir):  # function to load and format images
    images = []
    labels = []
    size = (IMG_WIDTH, IMG_HEIGHT)
    for folder in os.listdir(os.path.join(data_dir)):
        for filename in os.listdir(os.path.join(data_dir, str(folder))):  # for every images
            img = cv2.imread(os.path.join(data_dir, str(folder), filename))
            # print(img.shape)  # turn this on if you want to see every image's shape
            img = cv2.resize(img, size)  # resize the images
            if img is not None:
                images.append(img)
                labels.append(folder)
    return (images, labels)


def get_model():  # get the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            50, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(
            50, (2, 2), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


if __name__ == "__main__":  # run program
    main()
