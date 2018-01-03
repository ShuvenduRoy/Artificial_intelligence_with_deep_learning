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

""" Preprocessing the data"""
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# preprocessing labels
from keras.utils import to_categorical

train_lables = to_categorical(train_lables)
test_labels = to_categorical(test_labels)


"""training the model"""
# train
network.fit(train_images, train_lables, epochs=5, batch_size=128)

# test
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)