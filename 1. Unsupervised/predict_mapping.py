import time, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time, os, cv2
import tensorflow as tf
from glob import glob
from utils import *
from model_unsupervised_rn import *
from tqdm import tqdm
from tensorflow.keras.layers import Softmax



#Hyperparameters and input-output paths
#input_folder =  "Data/CV/mapping5"
#input_folder =  "Data/4x/mapping_jo2"
input_folder =  "Data/4x/train_512/images"
version_name = "rn2"
epoch = 50
N_C = 10                                # No. of clusters


#SIZE1 = 1440
#SIZE2 = 1920

SIZE1 = 512
SIZE2 = 512


test_path = input_folder+"/*.png"
test_images = sorted(glob(test_path))   # List of input images
#output_path  = "Results/Predicted_mapping1_"+version_name
output_path  = "Data/4x/train_512/deepcluster"
weights = "weights_"+version_name
print(output_path)

if os.path.isdir(output_path) is not True:
    os.mkdir(output_path)

def parse_data3(x):
    def _parse(x):
        x = read_image(x)
        return x
    x = tf.numpy_function(_parse, [x],  tf.float32)
    x.set_shape([SIZE1, SIZE2, 3])
    return x

def tf_dataset3(x, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.map(map_func=parse_data3)
    dataset = dataset.batch(batch)
    return dataset    

test = tf_dataset3(test_images, batch=1)

def prediction():
    Start = time.time()    
    for index, img in enumerate(test):
        name = os.path.basename(test_images[index])[:-4]
        image1,y,z= model.predict(img, batch_size=1, verbose=0, steps=None)
        Image= colour_code(image1[0])
        img = np.array(img)
        output =  output_path + f"/{name}_cluster.png"
        cv2.imwrite(output,Image)  
        #output =  output_path + f"/{name}_0_org.png"
        #cv2.imwrite(output,img[0,...,::-1]*255)      
    End = time.time()
    Time_taken = End - Start
    print(f"time_taken for evaluation {epoch} is {Time_taken/60:.2f} minutes " )




model = build_model(input_shape = (SIZE1,SIZE2,3),  num_classes = N_C)
model.load_weights(f"Weights/4x/{weights}_{epoch:03d}.hdf5")

def feature_weights(csv_name):
    w_3 = model.get_layer('rbf_layer').get_weights()
    w_64 = model.get_layer('rbf_layer_1').get_weights()
    w_128 = model.get_layer('rbf_layer_2').get_weights()
    w_256 = model.get_layer('rbf_layer_3').get_weights()
    w_512 = model.get_layer('rbf_layer_4').get_weights()
    w_3[1] = tf.where(w_3[1]<0.,0.,w_3[1])
    w_64[1] = tf.where(w_64[1]<0.,0.,w_64[1])
    w_128[1] = tf.where(w_128[1]<0.,0.,w_128[1])
    w_256[1] = tf.where(w_256[1]<0.,0.,w_256[1])
    w_512[1] = tf.where(w_512[1]<0.,0.,w_512[1])
    feature_values = K.concatenate((K.flatten(w_3[1]),K.flatten(w_64[1]),K.flatten(w_128[1]),K.flatten(w_256[1]),K.flatten(w_512[1])),axis=-1)
    with open(version_name+csv_name, "a") as f:
         np.savetxt(f, feature_values,fmt="%.4e",delimiter =",",newline=",")
         f.write("\n")        

#feature_weights("_10x_weights.csv")        
prediction()



