import tensorflow as tf

k = tf.placeholder(tf.float32)

#随机生成一个平均值变化的符合正态分布的数组，数组的长度为1000
moving_normal = tf.random_normal(shape=[1000],mean = 5*k)

#将生成的数组记录到直方图中去
tf.summary.histogram("moving_normal",moving_normal)

sess = tf.Session()

writer = tf.summary.FileWriter("histogram_logs")

summaries = tf.summary.merge_all()

N= 400

for x in [N]:
    k_val = x/float(N)
    summ = sess.run(summaries,feed_dict = {k:k_val})

    writer.add_summary(summ,global_step = x)
