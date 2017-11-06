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
    weight = tf.truncated_normal(shape=shape)
    return tf.Variable(weight)

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
b = Weight_Initial([10])

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
y = tf.nn.softmax(scores)

#交叉熵计算公式
loss = -tf.reduce_sum(tlbls*tf.log(y))

adam = tf.train.AdamOptimizer(0.001)
minimize = adam.minimize(loss)
#grad_b = adam.compute_gradients(loss,b)

out = tf.cast(tf.argmax(y,1),dtype=tf.float32)
#correct = tf.cast(tf.equal(tf.Variable(train_lbls),tf.cast(out,tf.int32)),tf.float32)
correct = tf.cast(tf.equal(train_lbls,out),tf.float32)
accuracy = tf.reduce_mean(correct)
##predictable
#te_input = tf.cast(tf.Variable(test_input),tf.float32)
##[10000,32,32,3]
#te_input = tf.reshape(te_input,shape=[10000,32,32,3])
#te_lbls = np.zeros(shape=[10000,10])
#for i in range(len(test_lbls)):
#    te_lbls[i][test_lbls[i]] = 1
#te_lbls = tf.Variable(te_lbls)

#test_out, = ConvLayers(te_input,te_lbls,W1,W2,W3,b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #b_val = sess.run(b)
    #print(b_val)
    for i in range(5):
        traindata = ReadDatasets("Images/cifar-10-batches-py/data_batch_"+(str)(i+1))
        #print(traindata.size)
        for j in range(200):
            tlbls_data = np.zeros([50,10])
            trainlabels = traindata.target[j*50:(j+1)*50]
            for i in range(50):
                tlbls_data[i][trainlabels[i]] = 1.0
            #tlbls = tf.cast(tf.Variable(tlbls),tf.float32)
            sess.run(minimize,feed_dict={data_input:traindata.data[j*50:(j+1)*50],train_lbls:trainlabels,tlbls:tlbls_data})
            if j%100 == 0:
                acc = sess.run(accuracy)
                print(acc)
    

    
    #sess.run(minimize)
    #test_pre = sess.run(grad_b)
    #print(test_pre)
    #acc = sess.run(accuracy)
    #print(acc)
    



##定义一个两行三列的矩阵
#matrix1 = tf.constant([[2,1,3],[5,2,4]])

##定义另一个三行两列的矩阵
#matrix2 = tf.constant([[1,5],[1,3],[2,1]])

##求两个矩阵的乘法值
#product = tf.matmul(matrix1,matrix2)

##定义一个标量的变量
#state = tf.Variable(2,None)

##计算一个矩阵与一个标量的乘积
#mult = tf.multiply(matrix1,state)

#mat_grad = tf.gradients(mult,matrix1)
#sta_grad = tf.gradients(mult,state)


##因为上面有变量定义，所以需要初始化变量
#initvar = tf.global_variables_initializer()

##启动会话,注意这个Session函数的大小写
#with tf.Session() as sess:

#    #设置使用特定的某个设备来执行，/gpu:0代表由第一个gpu来执行这个会话，/gpu:1等以此类推
#    #但当我输入10000时，并没有报错，可能是当找不到指定的gpu时，它会自动使用默认的设备计算。
#    #with tf.device("/gpu:0"):

#    #先运行初始化变量op
#    sess.run(initvar)
#    result = sess.run(product)
#    #当你需要获取多个op计算的结果时，调用多次run，传入你想获取的变量。
#    result2 = sess.run(mult)
#    print(result)
#    print("multiply:",result2)
#    #你也可以只调用一次run函数，获取多个值，如下所示
#    #result3 = sess.run([product,mult])
#    #print("result3:",result3[0],result3[1])

#    grad = sess.run([mat_grad,sta_grad])
#    print("gradient of mat:",grad[0])
#    print("gradient of state:",grad[1])

