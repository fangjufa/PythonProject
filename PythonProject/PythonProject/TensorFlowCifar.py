import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import read_images as ri
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def ReadDatasets(file):
    #train_cifarfile = "Images/cifar-10-batches-py/data_batch_"
    #for i in range(5):
    #train_cifardic = ri.unpickle(train_cifarfile + (str)(i))
    cifardic = ri.unpickle(file)
    #train_cifardic = ri.unpickle(train_cifarfile)
    datas = cifardic[b"data"]
    lbls = cifardic[b"labels"]
    return base.Dataset(data = datas,target=lbls)

#test_cifarfile = "Images/cifar-10-batches-py/data_batch_1"
#test_cifardic = ri.unpickle(test_cifarfile)
#test_input = test_cifardic[b"data"]
#test_lbls = test_cifardic[b"labels"]
#num = input.shape[0]

def Weight_Initial(shape):
    weight = tf.truncated_normal(shape=shape,stddev = 0.1)
    return tf.Variable(weight)

def Bias_Initial(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

#input = tf.cast(tf.Variable(train_input),tf.float32)
##[10000,32,32,3]
#input = tf.reshape(input,shape=[10000,32,32,3])
#tlbls = np.zeros([10000,10])
#for i in range(len(train_lbls)):
#    tlbls[i][train_lbls[i]] = 1.0
#tlbls = tf.cast(tf.Variable(tlbls),tf.float32)

data_input = tf.placeholder(dtype=tf.float32,shape=[None,3072])
input = tf.reshape(data_input,shape=[-1,32,32,3])
train_lbls = tf.placeholder(dtype=tf.float32)
tlbls = tf.placeholder(dtype=tf.float32)

#tlbls = tf.Variable(tlbls)
W1 = Weight_Initial([5,5,3,32])
W2 = Weight_Initial([5,5,32,64])
W3 = Weight_Initial([1600,10])
b = Bias_Initial([10])

#first convolution
#[10000,28,28,32]
conv1 = tf.nn.conv2d(input,W1,[1,1,1,1],padding = "VALID")
relu1 = tf.nn.relu(conv1)
#[10000,14,14,32]
maxpool1 = tf.nn.max_pool(relu1,[1,2,2,1],[1,2,2,1],padding="VALID")

#second convolution
#[10000,10,10,64]
conv2 = tf.nn.conv2d(maxpool1,W2,[1,1,1,1],"VALID")
relu2 = tf.nn.relu(conv2)
#[10000,5,5,64]
maxpool2 = tf.nn.max_pool(relu2,[1,2,2,1],[1,2,2,1],"VALID")

#full connected layer
fc = tf.reshape(maxpool2,shape=[-1,1600])
scores = tf.matmul(fc,W3) + b
#计算的y值，然后通过交叉熵，求loss，通过最小化loss求梯度。
#[10000,10]
y = tf.nn.softmax(scores,dim=0)

#交叉熵计算公式
loss = -tf.reduce_sum(tlbls*tf.log(y))

adam = tf.train.AdamOptimizer(0.001)
minimize = adam.minimize(loss)

out = tf.cast(tf.argmax(y,1),dtype=tf.float32)
#correct = tf.cast(tf.equal(tf.Variable(train_lbls),tf.cast(out,tf.int32)),tf.float32)
correct = tf.cast(tf.equal(train_lbls,out),tf.float32)
accuracy = tf.reduce_mean(correct)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        traindata = ReadDatasets("Images/cifar-10-batches-py/data_batch_"+(str)(i+1))
        #print(traindata.size)
        for j in range(200):            
            tlbls_data = np.zeros([50,10])
            trainlabels = traindata.target[j*50:(j+1)*50]
            for k in range(50):
                tlbls_data[k][trainlabels[k]] = 1.0
            #tlbls = tf.cast(tf.Variable(tlbls),tf.float32)
            if j%100 == 0:
                print(sess.run(b))
                acc = sess.run(accuracy,feed_dict={data_input:traindata.data[j*50:(j+1)*50],train_lbls:trainlabels,tlbls:tlbls_data})
                print(acc)
            sess.run(minimize,feed_dict={data_input:traindata.data[j*50:(j+1)*50],train_lbls:trainlabels,tlbls:tlbls_data})


