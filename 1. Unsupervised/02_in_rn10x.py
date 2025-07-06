import os , cv2, pickle
import numpy as np
from glob import glob
import scipy.ndimage
from tqdm import tqdm
from feat_rn import pred2, load_image
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Code for find initial feature based centers
# Loading the image
def load_image1(image_path):
    img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    img = img[:,:,:3]
    img = img/255
    return img

# Labelling the image
def labeling(cluster_centers, image):   
        image = image.reshape((-1,3))
        image = np.expand_dims(image , axis=0)
        image = np.repeat(image, K, axis =0)
        cluster_centers_exp = np.expand_dims(cluster_centers , axis=1)
        distance = np.sum(np.square(image - cluster_centers_exp),axis=-1)
        label = np.argmin(distance,axis=0)
        return label

# Finding the features center
def feature_centers(centers,feature,image, K):
        #print(feature.shape)
        c1 = feature.shape[-1]
        feature = feature.reshape((-1,c1))
        label = labeling(centers, image)
        res1 = []
        count, bins = np.histogram(label,bins=K, range=(0,K))
        for i in range(K):
            res2 = feature[label==i]
            res2 = np.sum(res2,axis=0)
            res1.append(res2)
        res1 = np.stack(res1,axis=0)    
        return res1, count



# Loading the RGB cluster centers    
file = open(f"kmeans_rgb10x", 'rb')
centers = pickle.load(file)
file.close()
print(centers)


filters = [3,64,256,512,1024]
image_path = "10x/*.png"
images = sorted(glob(image_path))
no_images = len(images)
K=10
no_layer = 5
new_centers=[]
total_results =0
total_counts =0
for img in tqdm (images, desc="Loading..."):
    features = pred2(img)
    image = load_image1(img)
    results= np.zeros((K,0))
    counts = 0
    for i in range(no_layer):
        result,count = feature_centers(centers,features[i],image, K)
        results= np.concatenate((results,result),axis=-1)
        counts = np.append(counts,count)
    total_results +=results 
    total_counts +=counts
total_counts = total_counts[1:]  
k=0
for i,j in enumerate(filters):
    y = total_counts[10*i:10*(i+ 1)]
    x= total_results[:,k:j+k]/y[:,np.newaxis]
    k +=j
    new_centers.append(x)
    

file = open(f"centers_rn_10x", 'wb')
pickle.dump(new_centers,file)
file.close()

       
