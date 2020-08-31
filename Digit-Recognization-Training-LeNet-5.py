#Importing Libraries
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Importing Model Framework Library
from keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical


(x_train, y_train),(x_test,y_test) = datasets.mnist.load_data()


print('x_train Shape: ',x_train.shape)
print('y_train Shape: ',y_train.shape)
print('x_test Shape: ',x_test.shape)
print('y_test Shape: ',y_test.shape)
print('Train samples: ',x_train.shape[0])
print('Test samples: ',x_test.shape[0])
print('Image Dimension: ',x_train[0].shape)


example = [0,9856,50587]

for _ in example:
    plt.figure()
    plt.imshow(x_train[_],cmap='Greys')
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print(y_train[_])


x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print('x_train shape:', x_train.shape)
print(x_train[0].shape, 'image shape')
print(y_train)
print(y_test)

#Convert class vectors to binary class matrices(i.e. One hot encoding)
num_classes = 10 #[0,1,2,3,4,5,6,7,8,9]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


print(y_train[0])
print(y_train[2])


#Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Pixel Value ranges form 0 to 255.(0- White & 255 - Black)
# As images are in 1 Channel i.e. GrayScale
#Converting pixel value to 0 to 1 (0- White & 1 - Black)
x_train /= 255
x_test /= 255


#Define LeNet Architecture
# Lenet Arch:
# Conv => Relu => Max Pool => Conv => Relu => Max Pool => FC => FC => Softmax
#In LeNet Architecture, Average Pooling is used instead of Max Pooling
#Activate Relu is used instead of Tanh.. for increasing accuarcy

def LeNet(input_shape,nb_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5,5), strides = (1,1), activation = "relu",input_shape = input_shape, padding = "same", name="Conv2D_layer1"))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid",name="Pool_layer1"))
    model.add(Conv2D(16, kernel_size = (5,5), strides = (1,1), activation = 'relu', padding='valid',name="Conv2D_layer2"))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), padding = 'valid',name="Pool_layer2"))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu',name="Dense_1"))
    model.add(Dense(84, activation = 'relu',name="Dense_2"))
    model.add(Dense(nb_classes, activation = 'softmax',name="Output_layer"))
    model.compile(optimizer = 'adam',loss = categorical_crossentropy, metrics = ['accuracy'])
    return model

model = LeNet(x_train[0].shape, num_classes)

print(model.summary())

model.fit(x_train,y= y_train, epochs=20,validation_data=(x_test, y_test))


class_names = [0,1,2,3,4,5,6,7,8,9]
prediction_values = model.predict_classes(x_test)

# set up the figure
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the images: each image is 28x28 pixels
for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i,:].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
  
    if prediction_values[i] == np.argmax(y_test[i]):
        # label the image with the blue text
        ax.text(0, 7, class_names[prediction_values[i]], color='blue')
    else:
        # label the image with the red text
        ax.text(0, 7, class_names[prediction_values[i]], color='red')


test_result = [0,985,1000]

for _ in test_result:
    plt.figure()
    plt.imshow(x_test[_].reshape(28,28),cmap='Greys')
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print('Expected : ',np.argmax(y_test[_],axis=-1))
    print('Prediction : ',prediction_values[_])


#Save your Model in .h5 format and delete it
model.save("Digit-mnist-LeNet.h5")
del model

