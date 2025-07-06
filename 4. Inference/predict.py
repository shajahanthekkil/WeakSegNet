import time, os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time, os, cv2
import tensorflow as tf
from glob import glob
from utils import *
from m_resunet import *
from tqdm import tqdm
from tensorflow.keras.layers import Softmax

#Hyperparameters and input-output paths
#version_name = "val_512"
version_name = "mapping_jo"
#version_name = "train_full"
epoch = 182
N_C = 5                                # No. of clusters
out_ver = "cv5"

if sys.argv[1:]:
   version_name = sys.argv[1]
   out_ver  = "cv"+version_name[-1]
   if sys.argv[2:]:
       epoch = int(sys.argv[2])

#SIZE1 = 1440
#SIZE2 = 1920

SIZE1 = 512
SIZE2 = 512


test_path = "Data/"+version_name+"/images/*"
test_path_gt = "Data/"+version_name+"/mask/*"
test_images = sorted(glob(test_path))   # List of input images
test_mask = sorted(glob(test_path_gt))   # List of input images
output_path  = "Results/Predicted_"+version_name
weights = "weights_"+out_ver
print(output_path)

if os.path.isdir(output_path) is not True:
    os.makedirs(output_path)

def parse_data3(x,y):
    def _parse(x,y):
        x = read_image(x)
        y = read_image(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([SIZE1, SIZE2, 3])
    y.set_shape([SIZE1, SIZE2, 3])       
    return x, y

def tf_dataset3(x,y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(map_func=parse_data3)
    dataset = dataset.batch(batch)
    return dataset    

test = tf_dataset3(test_images,test_mask, batch=1)


def prediction(acc_epoch):
    acc = np.zeros((0,N_C))
    Start = time.time()
    
    for index, (img,msk) in enumerate(test):
        name = os.path.basename(test_images[index])[:-4]
        print(name)
        image1= model.predict(img, batch_size=1, verbose=0, steps=None)
        Image= colour_code2(image1[0])
        img = np.array(img)
        msk = np.array(msk)
        output =  output_path + f"/{name}_{epoch:03d}_pred.png"
        cv2.imwrite(output,Image[...,::-1])  
        output =  output_path + f"/{name}_org.png"
        #cv2.imwrite(output,img[0,...,::-1]*255)
        output =  output_path + f"/{name}_gt.png"
        cv2.imwrite(output,msk[0,...,::-1]*255)
        image2 = one_hot(msk[0])
        acc = np.concatenate((acc,accuracy(image1, image2)),axis=0)
        

    acc = np.nanmean(acc,axis=0, keepdims= True)
    acc_epoch=np.concatenate((acc_epoch,acc),axis=0)
    End = time.time()
    Time_taken = End - Start
    print(f"time_taken for evaluation {epoch} is {Time_taken/60:.2f} minutes " )
    return acc_epoch


acc_epoch = np.zeros((0,N_C))

arch = ResUnetPlusPlus(input_size=None,no_classes=5)
model = arch.build_model()
model.load_weights(f"Weights/{weights}.{epoch:03d}.hdf5")
acc_epoch = prediction(acc_epoch)

