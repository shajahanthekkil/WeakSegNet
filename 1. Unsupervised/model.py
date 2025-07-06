import tensorflow as tf
import pickle
import numpy as np
from config import *
from rbflayer import RBFLayer, InitKMeans 
from keras.initializers import Initializer
from tensorflow.keras.layers import Input, Concatenate, Softmax, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add,Average,Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model

feature_name = "centers_rn_10x"		# initial centers for every layers
file = open(feature_name, 'rb')
centers_km = pickle.load(file)
file.close()


LoG = np.expand_dims(np.repeat(np.expand_dims(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),axis=-1),CLASSES,axis=-1),axis=-1)

class EdgeDetector(Initializer):
    def __init__(self, kernel1):
        self.kernal1 = kernel1   

    def __call__(self, shape, dtype=None):
        return self.kernal1





# Convolution blocks for the decoder part
def ConvBlock(X, n_filters, kernel_size=(3, 3)):
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(n_filters, kernel_size, strides = (1, 1) )(X)
    #X = BatchNormalization(axis = 3 )(X)
    X = Activation('relu')(X)
    return X

def conv_transpose_block(X, n_filters, kernel_size=(3, 3)):
    X = Conv2DTranspose(n_filters, kernel_size, strides=(2, 2), padding='same',output_padding= (1,1), activation=None)(X)
    X = BatchNormalization(axis = 3 )(X)
    X = Activation('relu')(X)
    return X

# Model Construction

def build_model(input_shape = (SIZE,SIZE,3),  num_classes = CLASSES):
    # Loading a VGG16 model pretrained on Imagenet dataset
    model_rn = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape)
    X0 = model_rn.input
    X1 = model_rn.get_layer('conv1_relu').output	# output from the second layer 64 features
    X2 = model_rn.get_layer('conv2_block3_out').output	# output from the second layer 128 features
    X3 = model_rn.get_layer('conv3_block4_out').output	# output from the second layer 256 features
    X4 = model_rn.get_layer('conv4_block6_out').output	# output from the second layer 256 features

    Y0, Z0 = RBFLayer(CLASSES, FILTERS_LAYER[0], initializer=InitKMeans(centers_km[0]))(X0) 	# RBF Layer for clustering with kmeans initialization
    Y1, Z1 = RBFLayer(CLASSES, FILTERS_LAYER[1], initializer=InitKMeans(centers_km[1]))(X1)	# RBF Layer for clustering with kmeans initialization
    Y2, Z2 = RBFLayer(CLASSES, FILTERS_LAYER[2], initializer=InitKMeans(centers_km[2]))(X2)  	# RBF Layer for clustering with kmeans initialization
    Y3, Z3 = RBFLayer(CLASSES, FILTERS_LAYER[3], initializer=InitKMeans(centers_km[3]))(X3)	# RBF Layer for clustering with kmeans initialization
    Y4, Z4 = RBFLayer(CLASSES, FILTERS_LAYER[4], initializer=InitKMeans(centers_km[4]))(X4) 	# RBF Layer for clustering with kmeans initialization

    Y1 = UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(Y1)
    Y2 = UpSampling2D(size=(4, 4), data_format=None, interpolation="bilinear")(Y2)
    Y3 = UpSampling2D(size=(8, 8), data_format=None, interpolation="bilinear")(Y3)
    Y4 = UpSampling2D(size=(16, 16), data_format=None, interpolation="bilinear")(Y4)
    
    W = Add()([Y0, Y1, Y2, Y3, Y4])
    # Regularization Model
    X = Softmax(axis=-1)(W)
    X = tf.keras.layers.DepthwiseConv2D((3,3),padding='same',use_bias=False,depthwise_initializer=EdgeDetector(LoG))(X)
    X = tf.math.abs(X)


    # decoder part
    X5 = conv_transpose_block(Z4, 512)
    X6 = Concatenate(axis=-1)([X5, Z3]) 
    X7 = conv_transpose_block(X6, 256)
    X8 = Concatenate(axis=-1)([X7, Z2]) 
    X9 = conv_transpose_block(X8, 64)
    X10 = Concatenate(axis=-1)([X9, Z1]) 
    X11 = conv_transpose_block(X10, 64)
    Y = ConvBlock(X11, 3)
    #model_label = Model(model_rn.input,y)		# The full network
        
    model = Model(model_rn.input,[W,X,Y])		# The cluster part of the model
    model.get_layer('depthwise_conv2d').trainable = False

    return model 
        
    
