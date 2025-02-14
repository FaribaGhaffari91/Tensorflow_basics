import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def house_price_prediction(X, Y):
    """
    This function is designed to build a simple neural network 
    to predict the house price.
    Note that since it is just a sample, the performance is not good at all (it is just for test!!!)
    """   
    #model definition
    model = tf.keras.Sequential([
        tf.keras.Input(shape = (1,)), # the NN takes one input
        tf.keras.layers.Dense(units = 1) # NN has one unit and gives one output
    ])

    model.compile(optimizer = 'adam' , loss = 'mean_squared_error') # 

    print(model.summary())

    #fit the model
    model.fit(X, Y, epochs = 1000)

    return model


def dataset_generation():
    """
    This function creates the dataset. To do this we are using the 
    real housing dataset downloaded from:
    Note that, since I am going to design a simple NN, which uses only one featur for price prediction, 
    only one column of the house price prediction dataset will be extracted as X and the price as Y.
    """ 

    dataset = pd.read_csv("./Housing.csv")
    columns = dataset.columns
    print(columns)
    data = dataset[['area' , 'price']]

    data_array = np.array(data, dtype = float)

    return data_array

def main():

    dataset = dataset_generation()
    train , test = train_test_split(dataset, test_size= 0.2)
    X_train = np.array(train[:,0], dtype = float)
    Y_train = np.array(train[:, 1]/10000, dtype=float) # devision is for normalizing data and bring X , Y in same range

    model = house_price_prediction(X = X_train, Y = Y_train)
    
    for i in range(len(test)):
      prediction = model.predict(np.array([test[i, 0]]), verbose=0)
      print(f"model prediction for {test[i,0]} number of bedrooms: {prediction.item() * 10000} (instead of {test[i, 1]})")
        

main()