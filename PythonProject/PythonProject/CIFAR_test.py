
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#从网站上读取mnist数据集，并将其存储到相对目录下mnist_train文件夹下。如果已经下载了就不会再下载。
#这个数据集是.gz的压缩文件，这个函数会自动将其解压缩，并将其分成train合test两部分。
#可以通过mnist.train和mnist.test的方式引用到训练集和测试集。
mnist = input_data.read_data_sets("mnist_train/",one_hot = True)
#image_cv = mnist.train.next_batch(1)[0][0]
#image_a = tf.reshape(image_cv,[28,28,1])
##print(mnist.train.next_batch(50)[0][0].shape)
#cv2.imshow("aasd",image_a)

#定义一个训练集。x是一个50000x784大小的张量(tensor)。
#它是将训练集的每一张图片都展成一维向量，总共有50000张图片。
#其实这里无所谓列向量或者行向量，但是一般python或者tensorflow都是列向量为主。
#784 = 28x28
x = tf.placeholder(tf.float32,[None,784])

#权重
W = tf.Variable(tf.zeros([784,10]))
#偏置量
b = tf.Variable(tf.zeros([10]))



'''第一层卷积'''
#卷积在每个5x5的patch中算出32个特征。
#卷积的权重张量形状是[5, 5, 1, 32]，
#前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
w_conv1 = weight_variable([5,5,1,32])  #filter
#对于每一个输出通道都有一个偏置量
b_conv1 = bias_variable([32])
#x_image的shape是[60000,28,28,1]
x_image = tf.reshape(x,[-1,28,28,1])

#卷积之后的shape是[60000,28,28,32],b_conv1的shape是[32],然后经过relu操作
#得到的shape仍然是[60000,28,28,32]
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,w_conv1,[1,1,1,1],padding = "SAME") + b_conv1)
#max_pool得到的shape是[60000,14,14,32]
h_pool1 = tf.nn.max_pool(h_conv1,[1,2,2,1],[1,2,2,1],padding = "SAME")
'''第一层卷积完'''

'''第二层卷积'''
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#relu操作是不会改变shape的。
#h_conv2的shape是[60000,14,14,64]
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2,[1,1,1,1],padding = "SAME") + b_conv2)
#h_pool2的shape是[60000,7,7,64]
h_pool2 = tf.nn.max_pool(h_conv2,[1,2,2,1],[1,2,2,1],padding = "SAME")

'''第二层卷积完'''

w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#h_pool2_flat shape是[60000,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#h_fc1的shape是[60000,1024]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

'''drop out'''
keep_prob = tf.placeholder("float")
#h_fc1_drop的shape是[60000,1024]
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

'''drop out完'''

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#预测值.这里tf.matmul函数的参数是有先后关系的，跟矩阵的左右关系是一样的。
#即matmul(a,b) = axb != matmul(b,a) = bxa.因为矩阵不符合乘法交换律。
#softmax操作里面的变量类型是[60000,10]，运算后，shape不变
y = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+ b_fc2)

#计算交叉熵
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#梯度递减算法。
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

with tf.Session() as sess:
#sess = tf.Session()
    sess.run(init)

    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
        
            train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            #print(batch[0].shape)
            #print(sess.run(h_fc1_drop,feed_dict={x:batch[0],keep_prob:1.0}))
            print("step %d,train accuracy %f"%(i,train_accuracy))
        sess.run(train_step,feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})

    print( "test accuracy %g",sess.run(accuracy,feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))#, keep_prob: 1.0}))
