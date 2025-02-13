import tensorflow as tf
import numpy as np

def hello_world_of_tensorflow(X, Y):
    """
    This function is the basic hello world function to show how to use tensorflow.
    It creates a neural networl with only one input and one output (there is no hidden layers).
    The purpose of this NN is to give some x as input and their reletive y, and we expect to 
    hopefully find a formula that can derive y from x for our next predictions.
    for example if we have the following (x,y) as input (for training):
    (1,2), (0,-1), (-2,-7), (5, 14)
    we can guess that y = 3x-1, so we expect out model learn this formula
    and for the input of x = 2 gives are nearly y = 5.
    More explaination on neural network is provided in this repository:
    """   
    # ------------ model definition --------------------------------
    model = tf.keras.Sequential([ # Sequential is used for defining the successive layers in tensorflow
        tf.keras.Input(shape = (1,)), # The input layer (or simply input of our NN) is a single number x
        tf.keras.layers.Dense(units = 1) #In this simple NN we have only one neuron defined by Dense (with one unit)
    ])
    
    # ----------compiling the model---------------
    # loss function and optimizers are used to find the parameters of NN
    model.compile(optimizer = "sgd" , loss = 'mean_squared_error') 


    #------------ model fitting ------------------
    # to fit the model with corresponding X, Y, 600 epochs means that the models will 
    # try the optimizer and loss functions for 600 times to find the best parameters 
    # to calculate y' most near to expected y.
    model.fit(X,Y, epochs = 600) 

    #our expected values are: (-7, -22) , (15, 44)
    # The reason that the coutput is not exact the 
    # same numbers is that tensorflow is using the real 
    # values of the numbers as float not integer to make it more precise.
    #plus, NN is always working with probabilities, the output values
    #are ery near to expected ones, but not exact the same
    print(f"model prediction for x = -7: {model.predict(np.array([-7]), verbose=0).item():.5f}")
    print(f"model prediction for x = 15: {model.predict(np.array([15]), verbose=0).item():.5f}")


# calling hello_world function
sample_x = np.array([1, 0, -2, 5, 7, 2, -1, -3, 4], dtype = float)
sample_y = np.array([2, -1, -7, 14, 20, 5, -4, -10, 11], dtype = float)
hello_world_of_tensorflow(sample_x, sample_y)