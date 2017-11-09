import sys
#sys.path.append('D:/PythonPackages/models-master/research/slim/nets')
sys.path.append('E:/models-master/research/slim/nets')
import inception_v4 as incept
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import read_images as ri

def ReadDatasets(file):
    cifardic = ri.unpickle(file)
    datas = cifardic[b"data"]
    lbls = cifardic[b"labels"]
    return base.Dataset(data = datas,target=lbls)

data_input = tf.placeholder(dtype=tf.float32,shape=[None,3072])
input = tf.reshape(data_input,shape=[-1,32,32,3])


onehot_lbls = tf.placeholder(dtype= tf.float32,shape=[None,10])

lables = tf.placeholder(dtype= tf.float32)

y = incept.inception_v4(input,10)

#交叉熵计算损失
loss = -tf.reduce_sum(onehot_lbls*tf.log(y))

#adam 计算梯度，并更新变量
adam_minimize = tf.train.AdamOptimizer(0.01).minimize(loss)

#然后计算准确度
calc_lbls = tf.arg_max(y,1)
correct = tf.cast(tf.equal(calc_lbls,lables),tf.float32)
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
                acc = sess.run(accuracy,feed_dict={data_input:traindata.data[j*50:(j+1)*50],lables:trainlabels,onehot_lbls:tlbls_data})
                print(acc)
            sess.run(minimize,feed_dict={input:traindata.data[j*50:(j+1)*50],lables:trainlabels,onehot_lbls:tlbls_data})


