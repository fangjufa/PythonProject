import sys
sys.path.append('D:/PythonPackages/models-master/research/slim/nets')
import inception_v4 as incept
import read_images as ri
import tensorflow as tf

def ReadDatasets(file):
    cifardic = ri.unpickle(file)
    #train_cifardic = ri.unpickle(train_cifarfile)
    datas = cifardic[b"data"]
    lbls = cifardic[b"labels"]
    return base.Dataset(data = datas,target=lbls)

#train_data = ReadDatasets("Images/cifar-10-batches-py/data_batch_0")
#输入，一张张图片。
input = tf.placeholder(dtype = tf.float32,shape=[None,32,32,3])
#shape为[None]，由一个个数字组成，表示在input当中，对应的图片是属于哪一个类别。
lables = tf.placeholder(dtype = tf.float32)
#每一行的内容是[0,0,1,0...]，即如果对应的图片类别是n,则该行的第n个数是1，其余全是0.
onehot_lbls = tf.placeholder(dtype = tf.float32,shape=[None,10])

#返回Softmax得到的值,shape为[None,10]，每一行的值是对应图片得到的对应类别的score，再求softmax。
y = incept.inception_v4(input,10)

#使用交叉熵计算loss。
loss = -tf.reduce_sum(onehot_lbls*tf.log(y))

adam = tf.train.AdamOptimizer(0.01)
minimize = adam.minimize(loss)

maxIdx_y = tf.argmax(y,axis=1)
correct = tf.cast(tf.equal(lables,maxIdx_y),tf.float32)
accuracy = tf.reduce_mean(correct)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_datasets = ReadDatasets("Images/cifar-10-batches-py/data_batch_0")

	for j in range(200):
		tlbls_data = np.zeros([50,10])
		trainlabels = train_datasets.target[j*50:(j+1)*50]
		for i in range(50):
			tlbls_data[i][trainlabels[i]] = 1.0
        #tlbls = tf.cast(tf.Variable(tlbls),tf.float32)
		sess.run(minimize,feed_dict={input:traindata.data[j*50:(j+1)*50],train_lbls:trainlabels,tlbls:tlbls_data})
		if j%100 == 0:
			acc = sess.run(accuracy)
			print(acc)






