# -*- coding: utf-8 -*-
"""使用CNN的多头绒泡菌图片处理
由于没有多头绒泡菌图片数据，暂时用MNIST数据代替来训练
by Yuqing Zhao, under guidance of Dr.Shanahan



Original file is located at
    https://colab.research.google.com/drive/1lYSCwcIzrwoQzVNrW57UUzyyaezVZ-qo


"""

import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""Set style"""

sns.set(style="whitegrid", font_scale=1.3)
matplotlib.rcParams["figure.figsize"] = (10, 8)
matplotlib.rcParams["legend.framealpha"] = 1
matplotlib.rcParams["legend.frameon"] = True

"""Just for the sake of reproducibility"""

np.random.seed(41)

"""# Data

In this tutorial we're going to use MNIST dataset with handwritten digits.

## MNIST overview

Download MNIST dataset. There is a special function in Keras for that purpose (because MNIST is extremely popular)
"""

# load MNIST data
from sklearn.model_selection import train_test_split
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
img_rows, img_cols = 28, 28
number_of_classes = 10
print(f"X before flatten train      shape: {X_train.shape}")
print(f"X before flatten validation shape: {X_valid.shape}")
print(f"X before flatten test       shape: {X_test.shape}")

X_train.shape

plt.figure(figsize=(12, 5))
for num, i in enumerate(np.random.choice(len(X_train), 10)):
    plt.subplot(2, 5, num + 1)
    plt.imshow(X_train[i], cmap="Greys_r")
    plt.axis("off")
    plt.title(str(y_train[i]))

"""Let's see objects are distributed among classes"""

x_bars, y_bars = np.unique(y_train, return_counts=True)
plt.bar(x_bars, y_bars)
plt.xlim([-1, 10])
plt.xticks(np.arange(0, 10))
plt.xlabel("Digit", fontsize=14)
plt.ylabel("Number of pics", fontsize=14)
plt.show()

"""As one can see, the task is pretty balanced

## Data preparation

First of all, let's predefine image parameters:
* **img_rows, img_cols** $-$ 2D dimension of a pictures; for MNIST it is $28 \times 28$
* **nb_classes** $-$ number of classes (digits in our case)
"""

img_rows, img_cols = 28, 28
nb_classes = 10



if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

"""Here we have grayscale image and thus the number of the channels is $1$. Here I used Tensorflow library with the corresponding order of dimensions"""

print('X_train shape:', X_train.shape)

"""Tensorflow prefers to work with $\text{float32}$ data type. So the next step is to cast data. Also let's have our data in $[0; 1]$ interval $-$ it's common choice for grayscale images."""

#Here we're going to use dense baseline models so we need to represent our data as 1-dimensional vectors
#So we flatten input images to 1D vectors 
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)

# Tensorflow/Keras prefers to work with float32 data type. 
# So the next step is to cast data. 
# Also let's have our data in [0; 1]interval; it's common choice for grayscale images.

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255
print(f"X train      shape: {X_train.shape}, {y_train.shape}")
print(f"X validation shape: {X_valid.shape},  {y_valid.shape}")
print(f"X test       shape: {X_test.shape},  {y_test.shape}")
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""## Setup the MNIST data
Setup the MNIST data. Here we use  **digits 0 to 9**.

Convert labels into [One-Hot Encoding](https://en.wikipedia.org/wiki/One-hot) because we're going to learn them through the softmax layer of CNN
"""

y_train = to_categorical(y_train, nb_classes)
y_valid = to_categorical(y_valid, nb_classes)
y_test  = to_categorical(y_test, nb_classes)

print(f"X train      shape: {X_train.shape}, {y_train.shape}")
print(f"X validation shape: {X_valid.shape},  {y_valid.shape}")
print(f"X test       shape: {X_test.shape},  {y_test.shape}")



"""# Dense baseline model

First of all, let's build MLP model and see how it performs
"""

model_dense = Sequential()

model_dense.add(Dense(128, input_shape=(img_rows * img_cols,), activation="relu"))
model_dense.add(Dropout(0.5))
model_dense.add(Dense(128, activation="relu"))
model_dense.add(Dropout(0.5))
model_dense.add(Dense(128, activation="relu"))
model_dense.add(Dropout(0.5))
model_dense.add(Dense(nb_classes, activation="softmax"))

"""Our model the the following architercture"""

model_dense.summary()

SVG(model_to_dot(model_dense, show_shapes=True).create(prog='dot', format='svg'))

"""Compile model"""

model_dense.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

"""## Training"""

import time
start = time.time()
epochs=20
hist = model_dense.fit(X_train.reshape((len(X_train), img_cols * img_rows)), y_train, 
                       validation_data = (X_valid, y_valid), 
                       epochs=epochs, batch_size=128)
end = time.time()
seconds_per_epoch = f"{(end - start)/epochs:.4}"
print(f"seconds_per_epoch: {seconds_per_epoch}")

"""## Evaluation"""

# Returns the loss value & metrics values for the model in test mode.
# CXE and accuracy
model_dense.evaluate(X_test,  y_test, verbose=1)

"""###  Visualize the learning, epoch by epoch"""

plt.figure(figsize=(20, 8))
plt.suptitle("Dense model training", fontsize=18)
plt.subplot(121)
plt.plot(hist.history["loss"], label="Train")
plt.plot(hist.history["val_loss"], label="Validation")
plt.grid("on")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Crossentropy", fontsize=14)
plt.legend(loc="upper right")
plt.title("Crossentropy CXE")
plt.subplot(122)
plt.plot(hist.history["acc"], label="Train")
plt.grid("on")
plt.plot(hist.history["val_acc"], label="Validation")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right")
plt.title("Accuracy")

#plt.ylim([0.88, 1.0]);

"""Table to store the results of the experiments

### Debugging (please use)
"""

'''
x = 3 
y =4

from IPython.core.debugger import Pdb as pdb;    pdb().set_trace() #breakpoint; dont forget to quit         

print(f"x: {x}")
'''

# add the result of this experiment to the log book
exp_name = "MLP-784-128-128-128-10" # experiment name
#del expLog
try:
    expLog
except NameError:
    expLog = pd.DataFrame(columns=["exp_name", "epoch (secs)", "Epochs", "Train CXE Loss", "Train Acc", "Validation CXE Loss", "Validation  Acc",
                    "Test CXE Loss", "Test  Accuracy"])
    
# Add a experiment results to the experiment log
model = model_dense
#from IPython.core.debugger import Pdb as pdb;    pdb().set_trace() #breakpoint; dont forget to quit         
expLog.loc[len(expLog)] = [f"{exp_name}", seconds_per_epoch, epochs] + list(np.round(np.reshape([model.evaluate(X_train, y_train, verbose=0), 
                   model.evaluate(X_valid, y_valid, verbose=0),
                   model.evaluate(X_test,  y_test, verbose=1)], -1), 3))
expLog



"""# Building CNN model

## Load MNIST data
"""

# load MNIST data
from sklearn.model_selection import train_test_split
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
img_rows, img_cols = 28, 28
number_of_classes = 10
print(f"X before flatten train      shape: {X_train.shape}")
print(f"X before flatten validation shape: {X_valid.shape}")
print(f"X before flatten test       shape: {X_test.shape}")

#Tensorflow prefers to work with float32 data type. 
#So the next step is to cast data. 
# Also let's have our data in [0; 1]interval; it's common choice for grayscale images.

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255
print(f"X train      shape: {X_train.shape}, {y_train.shape}")
print(f"X validation shape: {X_valid.shape},  {y_valid.shape}")
print(f"X test       shape: {X_test.shape},  {y_test.shape}")

"""### Input image Data reshaping 28x28x1

First of all, let's predefine image parameters:
* **img_rows, img_cols** $-$ 2D dimension of a pictures; for MNIST it is $28 \times 28$
* **nb_classes** $-$ number of classes (digits in our case)
"""

img_rows, img_cols = 28, 28
nb_classes = 10

"""Theano and Tensorflow both are tensor-based libraries. It means that all objects inside it, all inputs and outputs are **tensors**. One can treat tensor as a simple multidimensional array.

The thing that is different in Theano and Tensorflow is order of these dimensions inside tensor.

With Theano yo're going to have 4-dimensional tensor with the following dimensions: **(Objects, Channels, Image rows,Image columns)**. Assume that $\text{X_train}$ is our tensor. Then $\text{X_train}[0]$ gives you one trainig object - it is an image with few channels in general case. $\text{X_train}[0][0]$ gives you the first channel of the first object. And so on. The logic of tensors should be clear now.

In Tensorflow the order is the following: **(Objects, Image rows,Image columns, Channels)**

Thus we need to check what dimension order do we have and reshape our tensor in accordance with it:
"""

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    X_test  = X_test.reshape( X_test.shape[0],  1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

"""### Setup the MNIST Target data (OHE)
Setup the MNIST data. Here we use  **digits 0 to 9**.

Convert labels into [One-Hot Encoding](https://en.wikipedia.org/wiki/One-hot) because we're going to learn them through the softmax layer of CNN
"""

y_train = to_categorical(y_train, number_of_classes)
y_valid = to_categorical(y_valid, number_of_classes)
y_test  = to_categorical(y_test, number_of_classes)

print(f"X train      shape: {X_train.shape}, {y_train.shape}")
print(f"X validation shape: {X_valid.shape},  {y_valid.shape}")
print(f"X test       shape: {X_test.shape},  {y_test.shape}")



"""## Lay out the CNN Architecture

Now it's time to build the model step-by-step
"""

model_cnn = Sequential()

"""Out model is going to be *Sequential* which means that every new added layer will be automatically connected to the previous one.

Firstly, let's define hyperparameters of the network:
* **filters** $-$ number of filters (or kernels) to use in every layer; in fact this is the same as having multiple channels in the image
* **pool_size** $-$ size of the pooling window
* **kernel_size** $-$ size of the convolutional filters
"""

filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)

"""Now let's add first layer of the network. It is 2D Convolutional layer. Only unexplained thing here is *padding*. This is the parameter that defines how should we pad the data after applying convolutions. $\text{padding} = \text{'valid'}$ means that we're not going to pad images and the dimension of it is going to shrink from layer to layer."""

model_cnn.add(Convolution2D(filters=filters, 
                            kernel_size=kernel_size,
                            padding="valid",
                            input_shape=input_shape))

"""Next step is to add nonlinearity to enable our network to learn complex dependencies. We're going to use ReLU activation function because it is less exposed to vanishing gradient problem and faster to train."""

model_cnn.add(Activation('relu'))

"""Let's stack one more Convolution layer on top of that:"""

model_cnn.add(Convolution2D(filters=filters, 
                            kernel_size=kernel_size,
                            padding="valid"))
model_cnn.add(Activation('relu'))

"""Now it's time to apply pooling. Note that the strategies of combining convolutional and pooling layers may be different. For further details see [here](http://cs231n.stanford.edu/)"""

model_cnn.add(MaxPooling2D(pool_size=pool_size))

"""At this point we consider that we've already distinguish some meaningful features from the pictures. So it's time to classify them. For that purpose the common approach is to append fully-connected part. 

But before that we need to pull all the obtained feature into one vector so that one object has 1D-vector of features. It is done by means of $\text{Flatten}$ layer.
"""

model_cnn.add(Flatten())

"""Now let's add FC part with the [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) to avoid overfitting."""

model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(128, activation="relu"))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(nb_classes, activation="softmax"))

"""The final layer here is usual $\text{Softmax}$ with the number of classe. So as the output of the network we observe the probability of each class.

Now let's compile our model.
* **optimizer** $-$ here we use accelerated gradient descent algorithm with special adaptive way of choosing learning rate; for more details see this great [overview](http://sebastianruder.com/optimizing-gradient-descent/) of gradient descent optimization algorithms.
* **loss** $-$ usual choice for multiclass classification is softmax output layer in combination with categotical crossentropy loss function which is
$$
\mathcal{L}(\text{true}, \text{pred}) = -\sum_{j=1}^{k}\text{true}_j \cdot \log \{\text{pred}_j\}
$$
* **metrics** $-$ additional metrics that we're going to trace while training; it doesn't influence training process at all
"""

model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

"""Let's take a look at our final model now:"""

model_cnn.summary()

SVG(model_to_dot(model_cnn, show_shapes=True).create(prog='dot', format='svg'))

"""## Training

Training parameters are the following:
* nb_epoch $-$ number of epochs to train. here we choose 12; one may condiser using some stopping criterias
* **batch_size** $-$ parameter that controls how frequent do we update gradient; with $\text{batch_size}=1$ optimization is nothing but pure Stohastic Gradient Descent (update gradient after passing each one object); with $\text{batch_size}=\textit{number of objects}$ it will be usual Gradient Descent which updates gradient only after passing all objects. Choosing value between this two one can control speed and convergence of training process.
"""

batch_size = 128
epochs = 5

"""Train!"""

import time
start = time.time()
hist = model_cnn.fit(X_train, y_train, 
                     batch_size=batch_size, 
                     epochs=epochs,
                     validation_data=(X_valid, y_valid))
end = time.time()
seconds_per_epoch = f"{(end - start)/epochs:.4}"
print(f"seconds_per_epoch: {seconds_per_epoch}")

"""## Evaluation

Visualization of learning process:
"""

plt.figure(figsize=(20, 8))
plt.suptitle("CNN model training", fontsize=18)
plt.subplot(121)
plt.plot(hist.history["loss"], label="Train")
plt.plot(hist.history["val_loss"], label="Validation")
plt.grid("on")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Crossentropy", fontsize=14)
plt.title("Crossentropy CXE")
plt.legend(loc="upper right")
plt.subplot(122)
plt.plot(hist.history["acc"], label="Train")
plt.grid("on")
plt.plot(hist.history["val_acc"], label="Validation")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right")
plt.title("Accuracy")
plt.ylim([0.88, 1.0]);

"""Final evaluation of the model:"""

# add the result of this experiment to the log book
exp_name = "CNN-784_32x3x3-32_3x3-128-10" # experiment name
try:
    expLog
except NameError:
    expLog = pd.DataFrame(columns=["exp_name","epoch (secs)","Epochs","Train CXE Loss", "Train Acc", "Validation CXE Loss", "Validation  Acc",
                    "Test CXE Loss", "Test  Accuracy"])
    
# Add a experiment results to the experiment log
model = model_cnn
expLog.loc[len(expLog)] = [f"{exp_name}", seconds_per_epoch, epochs] + list(np.round(np.reshape([model.evaluate(X_train, y_train, verbose=0), 
                   model.evaluate(X_valid, y_valid, verbose=0),
                   model.evaluate(X_test,  y_test, verbose=1)], -1), 3))
expLog



