### Wesley Concepcion ###
### CID: welseyco ###
### FFR135: Artificial Neural Networks ###
### HP 2.1 Classification Challenge ###
### 7 Oct 2022 ###

# REFERENCES:
#https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras/notebook#3.-MNIST-dataset-
#https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial/notebook


import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential


def trainingSetLabels(y_train):
    individualLabels, count = np.unique(y_train, return_counts = True)
    print("Training Labels: ", dict(zip(individualLabels, count)))
    num_labels = len(np.unique(y_train))
    return num_labels


def visualize(x_train, y_train):
    index = np.random.randint(0, x_train.shape[0], size = 25)
    images = x_train[index]
    labels = y_train[index]
    plt.figure(figsize = (5, 5))
    for i in range(len(index)):
        plt.subplot(5, 5, i+1)
        image = images[i]
        plt.imshow(image, cmap = plt.get_cmap('gray'))
        plt.axis('off')

    return


def data_preprocessing(x_train, x_test, y_train, y_test):

    input_shape = (28, 28, 1)
    imgSize = x_train.shape[1]
    inputSize = imgSize * imgSize
    x_train = np.reshape(x_train, [-1, inputSize])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, inputSize])
    x_test = x_test.astype('float32') / 255
    y_train, y_test = encoding(y_train, y_test)

    return inputSize, x_train, x_test, y_train, y_test


def encoding(y_train, y_test):
    y_train = tf.one_hot(y_train.astype(np.int32), depth = 10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth = 10)
    return y_train, y_test


def DNN(x_train, x_test, y_train, y_test, inputSize, num_labels):
    #parameters
    batch_size = 128
    hidden_neurons = 256
    # learning_rate = 0.001
    dropout = 0.4
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=inputSize))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    #Evaluate the model
    loss, acc = model.evaluate(x_test, y_test, batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0*acc))
    return model


def runNetwork():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_labels = trainingSetLabels(y_train)

    inputSize, x_train, x_test, y_train, y_test = data_preprocessing(x_train, x_test, y_train, y_test)

    model = DNN(x_train, x_test, y_train, y_test, inputSize, num_labels)

    xTest2 = np.load('xTest2.npy')
    xTest2 = np.moveaxis(xTest2,3,0)
    imgSize = xTest2.shape[1]
    inputSize = imgSize * imgSize
    xTest2 = np.reshape(xTest2, [-1, inputSize])
    xTest2 = xTest2.astype('float32') / 255

    probability = model.predict(xTest2[:])
    prediction = np.argmax(probability, axis=1)
    prediction = prediction.astype(int)
    with open('classifications.csv', 'w') as f:
        writer = csv.writer(f)
        # for i in range(len(xTest2)):
        #     prediction[i] = int(prediction[i])

        writer.writerow(prediction)
        # print(probability[i], " => ", prediction[i])


if __name__ == "__main__":
    # visualize(x_train, y_train)
    runNetwork()


