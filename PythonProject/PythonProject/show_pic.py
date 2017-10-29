#import matplotlib.image as mpimg  
#import matplotlib.pyplot as plt  
#去除警告输出
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#import numpy as np
#import cv as cv2
  
##加载图像  
#filename = "curry.jpg"  
#image = mpimg.imread(filename)  
  
##创建tensorflow变量  
#x = tf.Variable(image,name='x')  
  
#model = tf.initialize_all_variables()  
  
#with tf.Session() as session:  
#    x = tf.transpose(x, perm=[1,0,2])     ##转置函数，通过该函数实现转置  
#    session.run(model)  
#    result = session.run(x)  
  
#plt.imshow(result)  
#plt.show()

#img = cv2.imread("curry.jpg")   
#cv2.namedWindow("Image")   
#cv2.imshow("Image", img)   
#cv2.waitKey (0)  
#cv2.destroyAllWindows()  

x = tf.constant(-7)
y = tf.constant([7,-3])
z = tf.constant([[-2,3],[6,5]])

xx = tf.nn.relu(x)
yy = tf.nn.relu(y)
zz = tf.nn.relu(z)

sess = tf.Session()
p = sess.run(zz)
print("p",p)