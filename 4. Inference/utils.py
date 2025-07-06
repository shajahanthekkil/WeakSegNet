from config_supervised import *
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras import layers
COLOR_ARRAY = np.array(COLOR_ARRAY)

def colour_code(image):
    x = np.argmin(image, axis = -1)
    res = COLOR_ARRAY[x]
    return res

def colour_code2(image):
    x = np.argmax(image, axis = -1)
    res = COLOR_ARRAY[x]
    return res

def accuracy(y_l,y_pred):
    y_l = np.argmin(y_l, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    y_l = y_l.flatten()
    y_pred = y_pred.flatten()
    matrix = confusion_matrix(y_l, y_pred,labels=LABELS)
    accuracy = matrix.diagonal()/matrix.sum(axis=1)
    accuracy = accuracy[np.newaxis,...]
    return accuracy
        
# Loss function for clustering
def cluster_loss(q):
    return K.sum(K.min(q,axis=-1))

def cluster_loss2(q):
    q1 = K.softmax(-q,axis=-1)
    q = q * q1
    return K.sum(q)

def recons_loss(x,y):
    loss_value = K.mean(K.mean( K.square(x-y),axis=-1))
    return loss_value

# Loss function for pseudo label, categorical cross entropy    
def label_loss(y_actual,y_predicted,weights):
    loss_value1 = -K.mean(K.sum( weights*y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    return loss_value1     
    
def label_loss_nw(y_actual,y_predicted):
    loss_value1 = -K.mean(K.sum(y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    return loss_value1     
    
def regularization_loss(r):
    return K.sum(r)

def custom_ce(y_actual,y_predicted):
    loss_value1 = -K.mean(K.sum( y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    return loss_value1 

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y      

def shuffling2(x):
    x = shuffle(x)
    return x

# Loading the input image
def load_image(image_path):
    img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    img = img[:,:,:3]
    img = img/255
    img = img[np.newaxis,...]
    return img
    

def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    image = image.astype(np.float32)
    return image

def one_hot(mask):
    semantic_map = []
    for colour in COLOR_ARRAY:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map


def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = one_hot(mask)
    mask = mask.astype(np.float32)
    return mask


    
def parse_data(x):
    def _parse(x):
        x = read_image(x)
        return x
    x = tf.numpy_function(_parse, [x],  tf.float32)
    x.set_shape([SIZE, SIZE, 3])
    return x

def tf_dataset(x, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.batch(batch)
    return dataset
    
def tf_dataset2(x, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.batch(batch)
    return dataset

def parse_data_supervised(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([SIZE, SIZE, 3])
    y.set_shape([SIZE, SIZE, num_classes])
    return x, y


def tf_dataset_supervised(x, y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(map_func=parse_data_supervised)
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.batch(batch)
    return dataset

def data_aug(image,seed1):
    data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical",seed=seed1),
  layers.RandomRotation(0.02, interpolation='nearest',seed=seed1),
  layers.RandomTranslation(height_factor=0.01, width_factor=0.01, fill_mode='reflect',
    interpolation='nearest', seed=seed1, fill_value=0.0),
  layers.RandomZoom(height_factor=(-0.1,0.1), width_factor=None, fill_mode='reflect',
    interpolation='nearest', seed=seed1, fill_value=0.0)])
    return data_augmentation(image)

def parse_data_aug(x):
    def _parse(x):
        seed1 = tf.random.uniform(shape=[],minval=0,maxval=2048)
        x = read_image(x)
        x = data_aug(x,seed1)
        return x
    x = tf.numpy_function(_parse, [x],  tf.float32)
    x.set_shape([SIZE, SIZE, 3])
    return x

def tf_dataset_aug(x, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.map(map_func=parse_data_aug)
    dataset = dataset.batch(batch)
    return dataset
