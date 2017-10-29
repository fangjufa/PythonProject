import cv2

import tensorflow as tf

image = cv2.imread("brick.png",3)

cv2.namedWindow('image', 0)  
cv2.imshow('image', image)

#初始化为TensorFlow的变量
x=tf.Variable(image,name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x,perm = [1,0,2])
    session.run(model)
    result = session.run(x)


cv2.namedWindow('result',0)
cv2.imshow('result',result)
