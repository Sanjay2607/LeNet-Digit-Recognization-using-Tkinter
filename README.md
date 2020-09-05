# LeNet-Digit-Recognization-using-Tkinter

LeNet is a convolutional neural network structure proposed by Yann LeCun et al. in 1998. In general, LeNet refers to lenet-5 and is a simple convolutional neural network.

LeNet possesses the basic units of the convolutional neural network such as convolution layer, pooling layer and full layer (F.C.). In LeNet, tanh is used as activation function and average pooling was used. But for better result, we are using Max Pooling Layer and Relu.


![LeNet Architecture](LeNet.PNG)


Layer1(Convolution Layer) : 6 kernels of 5x5, input image with shape=(28,28,1), stride = (1,1), padding = "same", activation = "relu".

Layer2(Max Pooling Layer) : pool_size = (2,2), strides = (2,2), padding = "valid".

Layer3(Convolution Layer) : 16 kernels of 5x5, stride = (1,1), padding = "valid", activation = "relu".

Layer4(Max Pooling Layer) : pool_size = (2,2), strides = (2,2), padding = "valid".

Flatten the layer

Layer5(Fully Connected Layer) : 120 Neurons, actiavtion = "relu"

Layer6(Fully Connected Layer) : 84 Neurons, activation = "relu"

Layer7(Output Layer) : 10 Neuron, Softmax each neuron representing output class.

optimizer = "adam"

loss = "categorical_crossentropy"

metrics = "accuracy"
