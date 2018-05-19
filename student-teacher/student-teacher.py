"""
Distilling the Knowledge in a Neural Network
http://arxiv.org/abs/1503.02531
"""

import keras
from keras.datasets import mnist
from keras.layers import Input, Embedding, Add, Dense, Lambda, Dropout, Activation
from keras.models import Model, Sequential
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size = 128
nb_classes = 10
nb_epoch = 3

# Define teacher model (the model that will learn form train data)
teacher = Sequential()
teacher.add(Dense(100, input_shape=(784,)))
teacher.add(Dense(10))
teacher.add(Activation('softmax'))
teacher.summary()

teacher.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = teacher.fit(X_train, Y_train, batch_size, nb_epoch, validation_data=(X_test, Y_test))

score = teacher.evaluate(X_train, Y_train, verbose=0)
print('Test Score: ', score[0])
print('Test accuracy: ', score[1])

# Freeze teacher model before training student model
for i in range(len(teacher.layers)):
    setattr(teacher.layers[i], 'trainable', False)

Y_train = np.zeros((60000, 10))

student = Sequential()
student.add(Dense(10, input_dim=784))
student.add(Activation('softmax'))
student.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])


def negativeActivation(x):
    return -x


negativeRight = Activation(negativeActivation)(student.output)
diff = Add()([teacher.output,negativeRight])

model = Model(inputs=[teacher.input, student.input], outputs=[diff])
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])

model.summary(line_length=150)
model.fit([X_train, X_train], [Y_train], batch_size=128, nb_epoch=5)


print(student.evaluate(X_test, Y_test))
