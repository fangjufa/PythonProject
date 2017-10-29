import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot = True)

Xtr,Ytr = mnist.train.next_batch(5000)
Xte,Yte = mnist.test.next_batch(200)

#将二维图像转化成一维数组。
Xtr = np.reshape(Xtr,newshape=(-1,28*28))
Xte = np.reshape(Xte,newshape=(-1,28*28))

#placeholder是什么意思，要查一下
xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])


distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.neg(xte))),reduction_indices=1)

pred = tf.arg_min(distance,0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
         nn_index= sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]}) #这是什么操作
         print( "Test" ,i,"Prediction:",np.argmax(Ytr[nn_index]),"True Class:",np.argmax(Yte[i]))
         if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
             accuracy += 1./len(Xte)

    print ("Done!")
    print ("Accuracy:",accuracy)
