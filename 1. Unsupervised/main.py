import time, os, cv2
import tensorflow as tf
from glob import glob
from utils import *
from model import *
from tqdm import tqdm
from tensorflow.keras.layers import Softmax
from config import *
import shutil

# Directories
version_name = "rn"
training_cluster_loss_csv= "training_cluster_loss_"+version_name+".csv"
training_label_loss_csv = "training_recon_loss_"+version_name+".csv"
evaluation_cluster_loss_csv= "evaluation_cluster_loss_"+version_name+".csv"
train_path = "./Data/train_512/*.png*"
test_path = "./Data/val_512/images/*.png"
test_gt = "./Data/val_512/mask/*.png"
output_path = "Output_"+version_name
model_name = "model_"+version_name+".h5"
weights_file = "weights_"+version_name
weight_folder = "Weights"

train_images = glob(train_path)
test_images = sorted(glob(test_path))
test_gt = sorted(glob(test_gt))
test = tf_dataset2(test_images, batch=1)

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(weight_folder):
    os.mkdir(weight_folder)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
weightage_recons = 10000
weight = (1/CLASSES) * tf.ones(CLASSES)              


model = build_model(input_shape = (SIZE,SIZE,3),  num_classes = CLASSES)
model.summary()
# back propogation for cluster update
@tf.function
def apply_gradient(optimizer, model, x):
    with tf.GradientTape() as tape:
        W, X, Y = model([x])                                           
        loss = cluster_loss2(W)                                      
        loss_reco = recons_loss(x, Y)                              
        loss_reg = regularization_loss(X)                           
        total_loss = loss + WEIGHTAGE_REGULARIZATION * loss_reg +  weightage_recons * loss_reco
    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss, loss_reg, loss_reco,total_loss                      

def decoder_only_training():
    Start = time.time()
    print(f"Epoch {epoch} started")
    for l in model.layers:
        l.trainable = False
    model.get_layer('conv2d_transpose').trainable = True
    model.get_layer('conv2d_transpose_1').trainable = True
    model.get_layer('conv2d_transpose_2').trainable = True
    model.get_layer('conv2d_transpose_3').trainable = True
    model.get_layer('concatenate').trainable = True
    model.get_layer('concatenate_1').trainable = True
    model.get_layer('concatenate_2').trainable = True
    model.get_layer('conv2d').trainable = True
    model.get_layer('batch_normalization').trainable = True
    # Back propogation for the DNN model using CE loss
    l_loss = 0
    pbar = tqdm(total=iterations, position=0,leave=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    for iteration in range(iterations):
        loss_c, loss_rc, loss_l,total_loss = apply_gradient(optimizer, model,train_set[iteration])                         
        l_loss += loss_l
        pbar.set_description("Image %s: Cluster Loss %0.3f, Cluster reg Loss %0.3f, Recons Loss %0.3f, Total loss %0.3f"%(int(iteration),float(loss_c),float(loss_rc), float(loss_l), float(total_loss))) 
        pbar.update()
    End = time.time()
    Time_taken = End - Start
    print(f"time_taken for epoch {epoch} completion is {Time_taken/60:.2f} minutes " )
    l_loss /= iterations
    print("PL loss", K.get_value(l_loss))

# Training the model
def train(loss_Cluster,loss_Regular,loss_Total,l_Loss):
    Start = time.time()
    weightage_recons = 1000

    print(f"Epoch {epoch} started")
    loss_cluster = 0
    loss_regular = 0
    loss_total = 0
    l_loss = 0
    pbar = tqdm(total=iterations, position=0,leave=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    for iteration in range(iterations):
        loss_c,loss_rc,loss_l ,total_loss = apply_gradient(optimizer, model, train_set[iteration])
        loss_cluster += loss_c
        loss_regular += loss_rc
        l_loss += loss_l
        loss_total += total_loss
        pbar.set_description("Image %s: Cluster Loss %0.3f, Cluster reg Loss %0.3f, Recons Loss %0.3f, Total loss %0.3f"%(int(iteration),float(loss_c),float(loss_rc), float(loss_l), float(total_loss))) 
        pbar.update()

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
        w_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w_3[1])) + tf.math.reduce_sum(tf.math.square(w_64[1])) + tf.math.reduce_sum(tf.math.square(w_128[1])) + tf.math.reduce_sum(tf.math.square(w_256[1]))  + tf.math.reduce_sum(tf.math.square(w_512[1])))
        w_3[1] /= w_norm
        w_64[1] /= w_norm
        w_128[1] /= w_norm
        w_256[1] /= w_norm
        w_512[1] /= w_norm


        model.get_layer('rbf_layer').set_weights(w_3)
        model.get_layer('rbf_layer_1').set_weights(w_64)
        model.get_layer('rbf_layer_2').set_weights(w_128)
        model.get_layer('rbf_layer_3').set_weights(w_256)
        model.get_layer('rbf_layer_4').set_weights(w_512)
            
        if (epoch % 20 == 0 and iteration==0):
            feature_values = K.concatenate((K.flatten(w_3[1]),K.flatten(w_64[1]),K.flatten(w_128[1]),K.flatten(w_256[1]),K.flatten(w_512[1])),axis=-1)
            with open(version_name+"_weights.csv", "a") as f:
                 np.savetxt(f, feature_values,fmt="%.4e",delimiter =",",newline=",")
                 f.write("\n")            
    End = time.time()
    Time_taken = End - Start
    print(f"time_taken for epoch {epoch} completion is {Time_taken/60:.2f} minutes " )
    loss_cluster /= iterations
    loss_regular /= iterations
    loss_total /= iterations
    l_loss /= iterations
    loss_Cluster.append(loss_cluster)
    loss_Regular.append(loss_regular)
    loss_Total.append(loss_total)
    l_Loss.append(l_loss)
    print("Cluster loss", K.get_value(loss_cluster))
    print("regularization loss",K.get_value(loss_regular))
    print("weighted loss",K.get_value(loss_total))
    print("PL loss", K.get_value(l_loss))
    return loss_Cluster,loss_Regular,loss_Total,l_Loss,loss_cluster

# evaluating the model 
def evaluation(eval_loss_Cluster ):
    eval_loss_cluster = 0 
    Start = time.time()
    for index, img in enumerate(test):
        image1,dummy,image2= model.predict(img, batch_size=1, verbose=0, steps=None)
        img = np.array(img)
        Image= colour_code(image1[0])
        output =  output_path + f"/{index:03d}_{epoch:03d}_cluster.png"
        cv2.imwrite(output,Image)  
        output =  output_path + f"/{index:03d}.png"
        cv2.imwrite(output,img[0,...,::-1]*255)
        output =  output_path + f"/{index:03d}_gt.png"
        shutil.copy(test_gt[index], output)
        output =  output_path + f"/{index:03d}_{epoch:03d}_recon.png"
        cv2.imwrite(output,image2[0,...,::-1]*255)  
        eval_loss_cluster += cluster_loss2(image1)
    eval_loss_cluster /= (index+1)
    print("Evaluation Cluster loss", K.get_value(eval_loss_cluster))
    eval_loss_Cluster.append(eval_loss_cluster) 
    model.save_weights(f"{weight_folder}/{weights_file}_{epoch:03d}.hdf5")

    End = time.time()
    Time_taken = End - Start
    print(f"time_taken for evaluation {epoch} is {Time_taken/60:.2f} minutes " )
    return eval_loss_Cluster , eval_loss_cluster 

# Training starts 
loss_Cluster = []
loss_Regular = []
loss_Total = []
l_Loss = []
eval_loss_Cluster =[]
epoch = 0
#model.load_weights(f"{weight_folder}/{weights_file}_{epoch:03d}.hdf5")

for epoch in range(0,EPOCHS):
    start = time.time()
    train_set = tf_dataset(train_images, batch=BATCH_SIZE)      
    train_set = list(train_set.as_numpy_iterator())            
    MAXITER = len(train_set)
    iterations = int(MAXITER/ BATCH_SIZE)
    print("Number of Training Images",MAXITER)

    # Decoder part is training by freezing the encoder part
    #if epoch < DECODER_ITERATIONS :
    if epoch < 20 :
        decoder_only_training()
    else:
    # Training entire model by after the decoder portion is trained
        #for l in model.layers:
        #    l.trainable = True
        #model.get_layer('depthwise_conv2d').trainable = False
        loss_Cluster,loss_Regular,loss_Total,l_Loss,loss_cluster =   train(loss_Cluster,loss_Regular,loss_Total,l_Loss)
       # with open("training_vgg_do.csv", "a") as f:  # append mode
       #     np.savetxt(f,[epoch,loss_cluster])
    # Evaluation
    #if (epoch % EVALUATION_INTERVAL==0):                             # model is evaluated at every EVALUATION_INTERVAL
    if (epoch % 20==0):                             # model is evaluated at every EVALUATION_INTERVAL
        print("evaluation")
        eval_loss_Cluster , eval_loss_cluster = evaluation(eval_loss_Cluster )
       # with open("evaluation_vgg_do.csv", "a") as f: # append mode
       #     np.savetxt(f,[epoch,eval_loss_cluster])
                     
np.savetxt(training_cluster_loss_csv, loss_Cluster,fmt='%2.5f', delimiter = ",")
np.savetxt(evaluation_cluster_loss_csv, eval_loss_Cluster,fmt='%2.5f', delimiter = ",")
np.savetxt(training_label_loss_csv, l_Loss,fmt='%2.5f', delimiter = ",")


end = time.time()
time_taken = end - start
print(f"time_taken is {time_taken/60:.2f} minutes " )


print("finished")
