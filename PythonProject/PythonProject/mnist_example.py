﻿"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     #���ز�����mnist����
x = tf.placeholder(tf.float32, [None, 784])                        #���������ռλ��
y_actual = tf.placeholder(tf.float32, shape=[None, 10])            #����ı�ǩռλ��

#����һ�����������ڳ�ʼ�����е�Ȩֵ W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#����һ�����������ڳ�ʼ�����е�ƫ���� b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#����һ�����������ڹ��������
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#����һ�����������ڹ����ػ���
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#��������
x_image = tf.reshape(x, [-1,28,28,1])         #ת����������shape,�Ա�������������
W_conv1 = weight_variable([5, 5, 1, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #��һ�������
h_pool1 = max_pool(h_conv1)                                  #��һ���ػ���

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #�ڶ��������
h_pool2 = max_pool(h_conv2)                                   #�ڶ����ػ���

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              #reshape������
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #��һ��ȫ���Ӳ�

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout��

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax��

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #������
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #�ݶ��½���
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #��ȷ�ȼ���
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:                  #ѵ��100�Σ���֤һ��
    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print('step',i,'training accuracy',train_acc)
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy",test_acc)