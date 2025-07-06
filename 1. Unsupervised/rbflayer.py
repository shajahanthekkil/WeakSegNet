from keras import backend as K
from tensorflow.keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import pickle





class InitKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, centers,max_iter=100):
        self.max_iter = max_iter
        self.centers = centers

       

    def __call__(self, shape, dtype=None):
        km =  self.centers
        return km

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])

# type checking to access elements of data correctly
        if type(self.X) == np.ndarray:
    	    return self.X[idx, :]
        elif type(self.X) == pd.core.frame.DataFrame:
    	    return self.X.iloc[idx, :]


class RBFLayer(Layer):

    def __init__(self, output_dim, scale_dim,initializer_scale_value =None, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.scale_dim = scale_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer

        if not initializer_scale_value:
            self.initializer_scale = tf.keras.initializers.Constant(1/(5*self.scale_dim))
            #self.initializer_scale = tf.keras.initializers.Constant(1/243)
        else:
            self.values_scale = initializer_scale_value
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[-1] ),
                                       initializer=self.initializer,
                                       trainable=True)
        self.scaling = self.add_weight(name='scaling',
                                       shape=(1,1,input_shape[-1],1),
                                       initializer=self.initializer_scale,
                                       trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        shape = K.int_shape(x)[-1]
        C = self.scaling * K.expand_dims(K.expand_dims(K.transpose(self.centers),axis=0),axis=0)
        H = self.scaling * K.repeat_elements(K.expand_dims(x), self.output_dim , axis = -1)
        I = H-C
        O = I**2
        return K.sum(O, axis=-2),H[...,0]
        #return K.sum(H**2, axis=-2)



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
