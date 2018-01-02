from keras.datasets import mnist
(train_images, train_lables), (test_images, test_labels) = mnist.load_data()

# defining the model architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# compiling the model
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])