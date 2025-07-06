import os
import cv2, time
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import scipy
from glob import glob
from tqdm import tqdm

# Code for finding rgb based kmeans as a initial centers

image_path = "10x/*.png"                      # Input Path
K = 10                                        # No. of clusters
C = 3   					# No. of channels
epsilon = 0.00001	
images = sorted(glob(image_path))		# Input images paths	
size = 512
no_images = len(images)		  
threshold_split =  0.95*size*size*no_images                                 
#threshold_split =  0.95*1920*1440*90                                 
print(threshold_split)
# Fuction for loading the image
def load_image(image):
        img = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        img = img/255
        return img
  
            
# Fuction for finding the label from the cluster centers        
def labeling(cluster_centers, image ,K):   
        image = image.reshape((-1,3))
        image = np.expand_dims(image , axis=0)
        image = np.repeat(image, K, axis =0)
        cluster_centers_exp = np.expand_dims(cluster_centers , axis=1)
        distance = np.sum(np.square(image - cluster_centers_exp),axis=-1)
        label = np.argmin(distance,axis=0)
        return label


# Function for updating the clusters
def update_centers(cluster_centers, mean_square, counts, image,K):
        label = labeling(initial_centers1, image, K)  
        image = image.reshape((-1,3))  
        count, bins = np.histogram(label,bins=K, range=(0,K))  
        counts+= count
        for i in range(K):
            cluster_centers_new = image[label==i]
            cluster_centers_new = np.sum(cluster_centers_new,axis=0)
            cluster_centers[i] += cluster_centers_new
            mean_square[i] += np.square(cluster_centers_new)
            
        return cluster_centers, counts, mean_square


# #######

initial_centers = np.abs(np.random.normal(loc=0.0, scale=0.1, size=(K,C)) )   # Random Gaussian for the initial centers



pickle_file = True
if pickle_file==True:
    file = open("kmeans_rgb10x", 'rb')
    final_centers= pickle.load(file)
    file.close()
else:  
    start = time.time()
    iterations = 10
    for j in range(iterations):
        initial_centers1 = np.copy(initial_centers)
        mean_square = np.square(initial_centers1)
        K = len(initial_centers1)
        initial_counts = np.ones(K)
        print("K",K)
        start1 = time.time()
        for i, image in enumerate(tqdm(images)):
            img = load_image(image)
            initial_centers, initial_counts, mean_square = update_centers(cluster_centers=initial_centers, mean_square=mean_square, counts=initial_counts, image=img , K=K)
        end = time.time()
        print(f"Time taken for the iteration {j+1} is {(end-start1)/60} minutes")
        print(initial_counts)
        initial_counts = np.expand_dims(initial_counts, axis=-1)
        biggest_cluster_count = np.max(initial_counts)
        biggest_cluster = np.argmax(initial_counts)
        
        initial_centers = initial_centers / initial_counts
        mean_square = mean_square / initial_counts
        final_centers = np.copy(initial_centers)
        print(final_centers)
        print(biggest_cluster_count)
        print(threshold_split)

        if biggest_cluster_count>threshold_split:
            sigma = (mean_square[biggest_cluster]) - np.square(initial_centers[biggest_cluster] )
            sigma = np.sqrt(sigma)            
            print("sigma",sigma)
            
            new_cluster1 = initial_centers[biggest_cluster] + epsilon * sigma
            new_cluster2 = initial_centers[biggest_cluster] - epsilon * sigma
            initial_centers[biggest_cluster] = new_cluster1
            initial_centers = initial_centers1[np.where(initial_counts>1)[0]]

            initial_centers = np.append(initial_centers,np.expand_dims(new_cluster2,axis=0),axis=0)

        
    file = open("kmeans_rgb10x", 'wb')
    pickle.dump(final_centers, file)
    file.close()
    end = time.time()
    print("timetaken = ", end-start)

print(final_centers)
K = len(final_centers)
     

        
for image in images:
        #print(image)
        img = load_image(image)
        label = labeling(final_centers, img, K)
        #new_color1 = np.array([[0,255,255],[255,255,255],[255,0,0],[0,255,0],[255,255,0],[0,0,255],[0,0,0],[255,0,255],[128,128,128],[0,128,128]])
        new_color1 = np.array([[255,0,0],[0,255,0],[255,0,255],[0,128,128],[255,255,0],[0,0,255],[0,0,0],[128,128,128],[0,255,255],[255,255,255]])  
        res = new_color1[label]
        result_image = res.reshape((512,512,3))
        output_path = image.replace('10x', '10x_kmeans')
        cv2.imwrite(output_path, result_image)
     
        
        
         


       
