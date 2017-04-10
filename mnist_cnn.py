# initially based on https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

# gets to ~0.9940 accuracy on test set after 12 epochs (~30s per epoch on NVidia GTX TITAN, 1060s per epoch on Intel i7)
# tested on ubuntu with Python 3, Keras version 1.2.2, tensorflow backend

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Merge

from keras.utils import np_utils
from keras.models import Model

#import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
num_classes = 10
nb_epochs = 12
nb_filters = 64
my_optimizer = 'adadelta'

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 60,000 training examples - each is of size 28 by 28
# Pixel values are 0 to 255. 0=white, 255=black 
# plot 4 images as grey scale
#plt.subplot(221)
#plt.imshow(x_train[0], cmap=plt.get_cmap('gray_r'))
#plt.subplot(222)
#plt.imshow(x_train[1], cmap=plt.get_cmap('gray_r'))
#plt.subplot(223)
#plt.imshow(x_train[2], cmap=plt.get_cmap('gray_r'))
#plt.subplot(224)
#plt.imshow(x_train[3], cmap=plt.get_cmap('gray_r'))
# show the plot
#plt.show()

# In 2D, "channels_last" assumes (rows, cols, channels) while "channels_first" assumes (channels, rows, cols). 

# there is only one channel here (levels of grey), so we need to create a dim here
# for color pictures we would have 3: RGB

# input_shape=(128, 128, 3) for channel_last

# option 'channels last'
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)	

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

my_max = np.amax(x_train)

x_train /= my_max # divide by 255 to have values between 0 and 1
x_test /= my_max

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# transforms integers labels into one-hot flags of length ncol
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# all theory is in here: http://cs231n.github.io/convolutional-networks/

my_input = Input(shape=input_shape, dtype='float32') # for some reason here it is important to let the second argument of shape blank

conv_1 = Convolution2D(nb_filters, 3, 3, # region size is (3, 3)
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (my_input)
# output is of dim  [(w - f + 2p) / s] + 1, where w is input size, f is filter size, s is stride, and p is amount of zero padding
# [28 - 3 + 2*0 / 1] + 1
pooled_conv_1 = MaxPooling2D(pool_size=(2,2)) (conv_1)
pooled_conv_1_dropped = Dropout(0.2) (pooled_conv_1)

conv_11 = Convolution2D(nb_filters, 3, 3, # region size is (3, 3)
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (pooled_conv_1_dropped)
pooled_conv_11 = MaxPooling2D(pool_size=(2,2)) (conv_11)
pooled_conv_11_dropped = Dropout(0.2) (pooled_conv_11)

pooled_conv_11_dropped_flat = Flatten()(pooled_conv_11_dropped)

# ====

conv_2 = Convolution2D(nb_filters, 4, 4, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (my_input)
pooled_conv_2 = MaxPooling2D(pool_size=(2,2)) (conv_2)
pooled_conv_2_dropped = Dropout(0.2) (pooled_conv_2)

conv_22 = Convolution2D(nb_filters, 4, 4, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (pooled_conv_2_dropped)
pooled_conv_22 = MaxPooling2D(pool_size=(2,2)) (conv_22)
pooled_conv_22_dropped = Dropout(0.2) (pooled_conv_22)

pooled_conv_22_dropped_flat = Flatten()(pooled_conv_22_dropped)

# ====

conv_3 = Convolution2D(nb_filters, 5, 5, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (my_input)
pooled_conv_3 = MaxPooling2D(pool_size=(2,2)) (conv_3)
pooled_conv_3_dropped = Dropout(0.2) (pooled_conv_3)

conv_33 = Convolution2D(nb_filters, 2, 2, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (pooled_conv_3_dropped)
pooled_conv_33 = MaxPooling2D(pool_size=(2,2)) (conv_33)
pooled_conv_33_dropped = Dropout(0.2) (pooled_conv_33)

pooled_conv_33_dropped_flat = Flatten()(pooled_conv_33_dropped)

# ====

# ====

conv_4 = Convolution2D(nb_filters, 6, 6, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (my_input)
pooled_conv_4 = MaxPooling2D(pool_size=(2,2)) (conv_4)
pooled_conv_4_dropped = Dropout(0.2) (pooled_conv_4)

conv_44 = Convolution2D(nb_filters, 6, 6, 
                       border_mode = 'valid',
					   activation = 'relu', 
                       #input_shape=input_shape
					  ) (pooled_conv_4_dropped)
pooled_conv_44 = MaxPooling2D(pool_size=(2,2)) (conv_44)
pooled_conv_44_dropped = Dropout(0.2) (pooled_conv_44)

pooled_conv_44_dropped_flat = Flatten()(pooled_conv_44_dropped)

# ====

merge = Merge(mode='concat') ([pooled_conv_11_dropped_flat,pooled_conv_22_dropped_flat,pooled_conv_33_dropped_flat,pooled_conv_44_dropped_flat])
merge_dropped = Dropout(0.2) (merge)

dense = Dense(128,
             activation='relu'
			) (merge_dropped)
dense_dropped = Dropout(0.2) (dense)

prob = Dense(output_dim = num_classes, # dimensionality of the output space
             activation='softmax'
			) (dense_dropped)

model = Model(my_input, prob)

print([layer.output_shape for layer in model.layers])

model.compile(loss='categorical_crossentropy',
			  optimizer=my_optimizer,
			  metrics=['accuracy'])

model.fit(x_train, 
		  y_train, 
		  batch_size = batch_size, 
		  nb_epoch = nb_epochs,
		  validation_data = (x_test, y_test)
		 )
		  