import tensorflow as tf
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("data/",one_hot = True)

#这里x还不是一个特定的值，是可变的。我们可以将其看成一个占位符。
#placeholder里面的两个参数，第一个描述的是类型，
#第二个描述的是这个数据，由任意个向量，每个向量有784维组成。None就是指任意数量的。
x = tf.placeholder(tf.float32,shape = [None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#这是预测值，即预测对应图片是什么数字
y = tf.nn.softmax(tf.matmul(x,W)+b)

#这是实际值，即对应图片是什么数字
y_ = tf.placeholder("float",[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#调用梯度降低算法，对交叉熵进行最小化的操作。是否返回值是回归的次数？
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#x = tf.Variable([[2,1,3]])

#y = tf.argmax(x,0)

init_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_var)

    #print("regression steps:",train_step)
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('mnist_logs', sess.graph)
    total_step = 0
    while training:
      total_step += 1
      session.run(training_op)
      if total_step % 100 == 0:
        summary_str = session.run(merged_summary_op)
        summary_writer.add_summary(summary_str, total_step)

    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
    
    #print(sess.run(y))
    
