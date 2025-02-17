import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def cloth_categorization():
    """
    This function is designed to build a simple neural network 
    to predict the category of clothing that the input belongs.
    """  
    train_set , train_label = dataset_generation('train')
    print(train_set.shape)

    # normalizing the pixel values (NN works better with normalized data)
    train_set = train_set/255

    model = tf.keras.Sequential([
        tf.keras.Input(shape =(1, 784)),
        tf.keras.layers.Dense(128, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
     
    model.compile(optimizer = tf._optimizers.Adam() , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(train_set, train_label, epochs = 100, callbacks = [epochCallback()])

    return model

def dataset_generation(state):
    """
    dataset: Fashion MNIST
    link to dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist
    More info about dataset: https://github.com/zalandoresearch/fashion-mnist
    Due to github size limit, I would not upload the dataset. After dowbloading 
    the dataset from kaggle and unzip it in the same folder as this py file, 
    it works as it is.

    """ 
    if state == 'train':
        train_dataset = pd.read_csv("datasets/fashion-mnist_train.csv")
        train_set_pd = train_dataset.iloc[:,1:]
        train_label_pd = train_dataset[['label']]
        train_set = np.array([train_set_pd], dtype=float)
        train_label = np.array([train_label_pd], dtype=float)
        return train_set, train_label

    elif state == 'test':
        test_dataset = pd.read_csv("datasets/fashion-mnist_test.csv")
        test_set_pd = test_dataset.iloc[:,1:]
        test_label_pd = test_dataset[['label']]
        test_set = np.array([test_set_pd], dtype=float)
        test_label = np.array([test_label_pd], dtype=float)
        return test_set, test_label
    else:
        return False

def main():
    test_index = 5500

    model = cloth_categorization()
    # normalizing the pixel values (the same normalization method used in training!!!)
    test_set , test_label = dataset_generation('test')
    test_set = test_set/255

    model.evaluate(test_set,test_label)
    
    #Printing the image (to have an over view of data)
    print(f"Label for the item number {test_index} is: {test_label[0][test_index]}")
    np.set_printoptions(linewidth=320)
    reshaped_image = test_set[0][test_index].reshape(28,28)
    print(f'\nImage looks like \n\n{reshaped_image}\n\n')

    plt.imshow(reshaped_image)
    plt.colorbar()
    plt.show()


class epochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if logs['loss'] < 0.5:
            print("loss is less than 0.5, training can be cancelled now!!!")
            self.model.stop_training = True    


main()