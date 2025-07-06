import cv2
import glob
import numpy as np
images =list(glob.iglob('5_*.png'))
for image in images:
    #print(image)
    img = cv2.imread(image,1)
    #print(img)
    image2 = image.replace('5_','6_')
    

    img[(np.all(img == np.array([255,0,0]),axis=-1))]=[0, 0, 255]
    cv2.imwrite(image2,img)
