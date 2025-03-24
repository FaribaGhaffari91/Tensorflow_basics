import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def cloth_categorization_convolutional_NN():
    """
    This function is designed to build a simple CNN 
    to predict the category of clothing that the input belongs.
    This model takes the image inputs in 28x28 pixels. since this can be large for
    process, the convolutional layers will be added to decrease the size. (In code I explained each one)
    """  
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train/255
    x_test = x_test/255

    model = tf.keras.Sequential([
        tf.keras.Input(shape = (28, 28, 1)),
        # first conv leyer
        # This layer first create a conv2D layer with a best-practice filter
        #Then it applies maxpool to choose the highest value of the pixel in a 2x2 dimention
        tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu), # first filter 
        tf.keras.layers.MaxPooling2D(2, 2), # after this line the image is already decreased by 4 in its size
        #second conv leyer
        # This layer first create another conv2D layer with a best-practice filter
        #Then it applies maxpool to choose the highest value of the pixel in a 2x2 dimention 
        tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu), # first filter 
        tf.keras.layers.MaxPooling2D(2, 2), # after this line the image is already decreased by 4 in its size
        # making 2D input in 1D
        tf.keras.layers.Flatten(),
        #From here, it is the same dense layer
        #Before reaching to this point a filter is applied to the image and decreased its size by 8
        tf.keras.layers.Dense(128, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

    model.summary()

    model.compile(optimizer = tf._optimizers.Adam() , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )
    model.fit(x_train, y_train, epochs = 5, callbacks = [callBack_CNN()])

    print("\nMODEL EVALUATION:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'test set accuracy: {test_accuracy}')
    print(f'test set loss: {test_loss}')

class callBack_CNN(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if logs['accuracy'] > 0.85:
            print("Accuracy is higher than 0.85, training can be cancelled now!!!")
            self.model.stop_training = True

cloth_categorization_convolutional_NN()