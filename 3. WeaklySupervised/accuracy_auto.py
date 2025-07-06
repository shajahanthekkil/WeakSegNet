from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score, recall_score
import csv
from glob import glob
import numpy as np
import cv2

version = "Test5"
#version = "mapping2"
K1 = 10
K2 = 4

def accuracy(y_l,y_pred,K=10):
    y_l = np.argmax(y_l, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    y_l = y_l.flatten()
    y_pred = y_pred.flatten()
    if K==10:
        matrix = confusion_matrix(y_l, y_pred,labels=[0,1,2,3,4,5,6,7,8,9])
        f1 = f1_score(y_l, y_pred,labels=[0,1,2,3,4,5,6,7,8,9],average=None)
        p1 = precision_score(y_l, y_pred,labels=[0,1,2,3,4,5,6,7,8,9],average=None)
        r1 = recall_score(y_l, y_pred,labels=[0,1,2,3,4,5,6,7,8,9],average=None)        
    else:
        matrix = confusion_matrix(y_l, y_pred,labels=[0,1,2,3]) 
        f1 = f1_score(y_l, y_pred,labels=[0,1,2,3],average=None)
        p1 = precision_score(y_l, y_pred,labels=[0,1,2,3],average=None)
        r1 = recall_score(y_l, y_pred,labels=[0,1,2,3],average=None)
    accuracy = matrix.diagonal()/matrix.sum(axis=1)
    accuracy = accuracy[np.newaxis,...]
    
    f1 = f1[np.newaxis,...]
    p1 = p1[np.newaxis,...]
    r1 = r1[np.newaxis,...]
    over_acc = np.nanmean(accuracy, axis=-1)
    return accuracy, over_acc, matrix, f1, p1,r1
    

label_values1 = np.array([[255,255,255],[255,0,0],[0,255,0],[0,165,255],[255,255,0],[0,0,255],[0,0,0],[255,0,255],[0,255,255],[128,128,128]])
label_values2 = np.array([[255,255,255],[0,0,255],[0,255,0],[255,0,255]])
label_values3 = np.array([[255,255,255],[0,0,255],[255,0,0],[0,255,0],[255,0,255]])

def one_hot(mask, label_values =label_values1 ):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map

def accuracy_folder(images_pl,images_gt, K=10):
    acc  = np.zeros((0,K))
    f1_sc  = np.zeros((0,K))
    p1_sc  = np.zeros((0,K))
    r1_sc  = np.zeros((0,K))
    matr  = np.zeros((K,K))
    ov_acc = []
    for index, item in enumerate(images_pl):
        gt = cv2.imread(item)
        image = cv2.imread(images_gt[index])
        if K==10:
            gt  = one_hot(gt, label_values1)
            image = one_hot(image,label_values3)
        elif K==4:
            gt  = one_hot(gt, label_values2)   
            image[np.all(image==[255,0,0],axis = -1)] = [0,0,255]
            image = one_hot(image,label_values2)                 
        print(index,"\n")
               
        accu, over_acc, matrix, f1, p1, r1 = accuracy(image,gt,K)  
        ov_acc.append(over_acc)
        acc = np.concatenate((acc,accu),axis=0)
        f1_sc  = np.concatenate((f1_sc,f1),axis=0)
        p1_sc  = np.concatenate((p1_sc,p1),axis=0)
        r1_sc  = np.concatenate((r1_sc,r1),axis=0)
        matr += matrix
    acc = np.nanmean(acc,axis=0, keepdims= False)
    f1_sc = np.nanmean(f1_sc,axis=0, keepdims= False)
    p1_sc = np.nanmean(p1_sc,axis=0, keepdims= False)
    r1_sc = np.nanmean(r1_sc,axis=0, keepdims= False)
    return np.mean(ov_acc,keepdims=True), acc, matr, f1_sc ,p1_sc, r1_sc
    
def colour_code(image, label_values):
    x = np.argmax(image, axis = -1)
    res = label_values[x]
    return res    

#header = ['W',	'B',	 'G',	 'O',	'C'	, 'R',	 'black',	 'M',	 'Y',	' Gray' , 'overall']


f = open(version+'_cluster1_accuracy_.csv', 'w')
writer = csv.writer(f)
input_fold1 = "Data/CV/"+version+"/mask/*_cluster*"
output_fold = "Data/CV/"+version+"/mask/*_gt*"    
images_gt = sorted(glob(output_fold))
images_pl = sorted(glob(input_fold1))
mapping = False
a = np.array([1,3,0,0,0,0,0,0,3,4])
#a = np.array([1,0,0,0,0,3,0,0,3,4])
print(a)
if mapping:
    ov, acu, mat, f1_sc, p1_sc, r1_sc = accuracy_folder(images_pl,images_gt,K=K1)
    a = np.argmax(mat,axis=0)
    print(a)

for index, item in enumerate(images_pl):
    print(item)
    gt = cv2.imread(item)
    gt1 = np.copy(gt)
    gt1 = one_hot(gt1, label_values1 )
    gt1 = colour_code(gt1,a)
    print(np.unique(gt1))
    gt1 = label_values3[gt1]
    print(np.unique(np.reshape(gt1,(-1,3)),axis=0))
    item2 = item.replace("cluster.","c1.")
    print(item2)
    print("\n")
    cv2.imwrite(item2,gt1)

print(a)  
header = ['Background (W)', 'Cell (R)', 'Cytoplasm (G)',	'Isolated(M)'	, 'overall_accuracy' , 'Metric']    
writer.writerow(header)
input_fold2 = "Data/CV/"+version+"/mask/*_c1*"  
images_pl = sorted(glob(input_fold2))
ov, acu, mat, f1_sc, p1_sc, r1_sc = accuracy_folder(images_pl,images_gt,K=K2)
print(mat.T)
print(ov, acu, f1_sc, p1_sc, r1_sc)
result = np.append( acu, ov[0] , axis = -1)
writer.writerow(result)
writer.writerow(f1_sc)
writer.writerow(p1_sc)
writer.writerow(r1_sc)
f.close()


