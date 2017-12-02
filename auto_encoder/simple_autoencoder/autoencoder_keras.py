from keras.layers import Input, Dense
from keras.models import Model

# this is the size of encoded dimenstion
encoding_dim = 32  # which have compression factor of 24.5 (784/32)

# input placeholder
input_img = Input(shape=(784, ))

# encoded representation of input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# decoded representation of input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]

# create decoder model
decoder = Model(encoded_input, decoded_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
