import tensorflow as tf
import numpy as np

def house_price_prediction():
    """
    This function is designed to build a simple neural network 
    to predict the house price.
    """   
    X, Y = dataset_generation(10)

    #model definition
    model = tf.keras.Sequential([
        tf.keras.Input(shape = (1,)), # the NN takes one input
        tf.keras.layers.Dense(units = 1) # NN has one unit and gives one output
    ])

    model.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

    print(model.summary())

    #fit the model
    model.fit(X, Y, epochs = 1000)

    return model


def dataset_generation(sample_num):
    """
    This function generates dataset using the following formula:
    -one bedroom appartment is 100K
    -each bedroom adds 50K
    """ 
    X = np.array([num_bedroom+1 for num_bedroom in range (sample_num)], dtype = float)
    Y = np.array([1 + (i-1) * 0.5 for i in X], dtype = float)
    return X , Y

def main():

    x_test = np.array([15, 11, 20], dtype= float)
    print(x_test[0])

    model = house_price_prediction()
    for i in range(len(x_test)):
        y_test = model.predict(np.array([x_test[i]]), verbose=0)
        print(f"model prediction for {x_test[i]} number of bedrooms: {y_test.item()} hundreds of thusand")
        

main()