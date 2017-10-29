import numpy as np
import tensorflow as tf


#初始化theta向量，它是一个二维的向量(theta0,theta1)
#该变量如何初始化
theta = tf.Variable(tf.zeros([2]))

#步长
alpha = 0.01

result = 1000;

#ai = [1,xi],这里有未知条数据，所以用None表示第一维的数。
ai = [1,123]#[[1,123],[1,150],[1,87],[1,102]]# tf.placeholder("float", [4,2])

y = [250,320]#,160,220]

#关于theta的导数(derative)
j_theta_der = tf.reduce_sum((theta*ai-y)*ai,1)

j_theta = 0.5*tf.reduce_sum((theta*ai-y)**2)



init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(100):
    res = sess.run(j_theta)
    #如果当前求得的值比上一次求得的值还小，则更新这个值。
    if res < result:
        res = result
        #theta = 
    else:
        print(theta)
        
    

    
#sess.run(

#tf.reduce_sum(
#a = tf.reduce_sum(
#init = tf.initialize_all_varialbles()

#ss = tf.Session
#ss.run(init)
