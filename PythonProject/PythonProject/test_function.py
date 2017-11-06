
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''测试truncated_normal函数'''
#import tensorflow as tf
##truncated_normal函数生成一个遵从正态分布的数组。shape参数可以传rank为1的数组，也可以传rank为n的数组。
##可以指定均值和标准差。

##生成一个长为10的一维数组，其均值是默认的0，标准差是0.1。
##这样的话，它所生成的所有10个数都是在均值加正负两个标准差范围内。如果生成的数不在这个范围内，就会遗弃掉这个数，重新生成。
#a = tf.truncated_normal([10],stddev = 0.1)

##生成一个5行10列的二维数组，均值是5，标准差是0.2
##其每一列的数都遵循均值和标准差的设定。
#b = tf.truncated_normal([5,10],mean=5,stddev = 0.2)
#with tf.Session() as sess:
#    print(sess.run(a))
''' end '''
#a = tf.constant(0.1,shape=[3,4])
#b = tf.Variable(a)

#init = tf.global_variables_initializer()

#with tf.Session() as sess:
#    sess.run(init)
#    print(sess.run(b))

#input = tf.Variable(tf.random_normal([1,3,3,3]))
#filter = tf.Variable(tf.random_normal([1,1,3,1]))

#[1,3,3,3]
#input = tf.constant([[[[2,2,1],[1,2,2],[2,1,1]],[[1,2,1],[1,2,2],[2,1,1]],[[1,2,1],[1,2,2],[2,1,1]]]],tf.float32)

#filter = tf.constant([[[[2,1,2,2],[2,2,1,2],[1,1,1,1]]]],tf.float32) #[1,1,3,3]

#op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')


'''测试tensorflow对于张量加减法的规则'''
#import tensorflow as tf
#a = tf.constant([[1],[2]])#[2,1]
#b = tf.constant([[1],[2]])  #[3]
#c = tf.constant([1])      #[1]
#d = tf.constant([1,1,1,1])#[4]
#e = tf.constant([[1,2],[3,4]])
#f = tf.constant([[1,2],[3,4],[5,6]])

#g = tf.reshape(f,[-1,2,2])

#init = tf.global_variables_initializer()

#sess = tf.Session()
#sess.run(init)
##print("input:",sess.run(input))
##print("filter:",sess.run(filter))
##print("a+b:",sess.run(a+b))
##print("a+c:",sess.run(a+c))
##print("a+d:",sess.run(a+d))
##print("a+e:",sess.run(a+e))
##print("a+f:",sess.run(a+f))
#print(g.shape())
''' end'''

'''测试卷积函数'''
#import tensorflow as tf
#def truncated_variable(shape):
#    initial = tf.truncated_normal(shape,stddev = 0.1)
#    return tf.Variable(initial)

#source = truncated_variable([10,32,32,1])

#filter = truncated_variable([5,5,1,32])

#conv_img = tf.nn.conv2d(source,filter,[1,1,1,1],padding = "SAME")

#init = tf.global_variables_initializer()

#with tf.Session() as sess:
#    sess.run(init)
#    print(sess.run(conv_img).shape)  #result[1,32,32,32]

'''end '''

'''测试reshape时如果维数不匹配，会不会报错'''
#import tensorflow as tf
#def truncated_variable(shape):
#    initial = tf.truncated_normal(shape,stddev = 0.1)
#    return tf.Variable(initial)

#source= truncated_variable([60000,8,8,64])
#result = tf.reshape(source,[-1,7*7*64])


#with tf.Session() as sess:
#    sess.run(init)
    #print(sess.run(result).shape)
    #print(sess.run(x_a,feed_dict={x:}).shape)

'''end'''


'''测试matmul函数'''
#import tensorflow as tf
#def truncated_variable(shape,mean = 0,stddev = 0.1):
#    initial = tf.truncated_normal(shape,mean,stddev)
#    return tf.Variable(initial)

#a = truncated_variable([2,3],mean = 1)
#b = truncated_variable([2,2],mean = 1)
#result = tf.matmul(b,a)

#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    print(sess.run(result).shape) #得到的结果是[2,3],证明tensorflow是行向量优先的。

'''end'''

'''测试dropout函数'''
#import tensorflow as tf
##经测得，不等于0的个数不是一定的，dropout是不为零的概率，不为零的个数大概在这个范围内。
#dropout = tf.placeholder(tf.float32)
#x = tf.Variable(tf.ones([10, 10]))
#y = tf.nn.dropout(x, dropout)

#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)

#result = sess.run(y, feed_dict = {dropout: 0.2})

#print( result)

#count = 0

#for i in range(10):
#    for j in range(10):
#        if result[i,j] != 0:
#            count = count+1

#print("count:",count)
    
'''end'''

''' 测试softmax函数'''
#import tensorflow as tf
#得出结果，将tensor张量的每一个分量做soft max运算。
#t = tf.constant([0.,1.,2.])

#exp = tf.exp(t)
#reduce = tf.reduce_sum(exp,-1)
#softmax = tf.nn.softmax(t)

#sess = tf.Session()
#print(sess.run(exp))
#print(sess.run(reduce))

#result = sess.run(softmax)
#for i in range(3):
#    print(result[i])

'''end'''

'''测试向量与一维常量的线性运算'''
#x = np.linspace(0,5,10)
#y = x * 0.5 + 10
#print(x)
#print(y)

#file_path = "D:/PythonProjects/TestPython/TestPython/TestPython/UnityLoader.txt";
#file = open(file_path,"r")

#str = file.read()
#print(str)

'''end'''

'''测试函数的局部变量和全局变量'''
#x = 0.5
#y = 0.4
#def func(y):
#    global x
#    x += 1
#    y = x + 0.6
#    print(x)
#    y += 1
#    print(y)

#func(0.4)
#print(x)

#x = [0.2,1,0.3]
#y = np.add(x, - 0.2)
#print(y)
'''end'''

'''np.all的用法'''
#arr = np.zeros(9)
#arr[4] = 2

#if np.all(arr == 0):
#    print("allzero")

#else:
#    print("not")

#arr = np.zeros(20)
##设置数组arr从下标为0到下标为9这10个数都为5.
#arr[0:10] = 5
#print(arr[8])

'''end'''
'''测试字符串拼接方法'''
#file = "mnist_train/"
#for i in range(10):
#    imgi = file + "train%d"%i
#    for j in range(10):
#        print(imgi + "/train%d_%d.jpg"%(i,j))
        #num_img = cv2.imread()
'''测试数组与标量,矩阵与矩阵的运算'''
#arr_a = [2,3,4]
#arr_b = [1,4,5]
#print(np.multiply(arr_a,1/2))
#print( np.dot(arr_a,arr_b))
#mat_a = np.array([[2,3,5],[4,1,3],[2,3,6]])
#mat_b = [[3,1,2],[2,1,6],[2,2,7]]
##要进行这个操作，mat_a一定要是numpy的数组变量，python的数组变量会报错。
##即定义的时候，要加上np.array().
##这里的0:2,是从第几列开始到第几列结束。是左闭右开区间。
##即0:2取第0行和第一行，1:3取第1、2两行。
#print(mat_a[0:2,0:2].shape)
#print(mat_a[0,:].ndim)
#print(mat_a[0][2])

#mat_c = np.array([[[2,4],[5,2]],[[4,1],[3,5]]])
#print(np.sum(mat_c,axis = 0))

#multiply函数是各个对应元素的乘积。
#axis = 0,加总列的所有元素，axis=1，加总行的所有元素。
#print(np.sum(np.multiply(mat_a,mat_b),axis = 1))
#vec_b = [5,5]
#vec_c = [5,5.0]
#print("Is vec_b equal to vec_c?", vec_b == vec_c)
#vec_b = np.add(vec_b, vec_c)
#print(vec_b)
##This is not work,use the one below.
##print(vec_c/5)
#print("Multyply a scalar:", np.multiply(vec_c,1/5))
##This is not work
##np.all(vec_c = 0)
#print(vec_c)
##数学意义上的矩阵与矩阵相乘，矩阵与标量相乘，都是用dot函数。
#print(np.dot(mat_a,vec_b))

'''end'''

'''测试自定义类的使用'''
#import lossfunction as lf

#LF = lf.LossFuntion()
#LF.A("Test class")

'''end'''
'''测试绘图更新'''

#x = np.linspace(0, 6*np.pi, 100)
#y = np.sin(x)

## You probably won't need this if you're not embedding things in a tkinter plot...
## and it should be put above the plt.figure() statement.
#plt.ion()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

#for phase in np.linspace(0, 10*np.pi, 100):
#    line1.set_ydata(np.sin(x + phase))
#    fig.canvas.draw()

#plt.close()

'''end'''
'''测试如何获取数组的维数'''
#mat_a = np.zeros([2,2,2,2])
##a = mat_a.shape
#print(mat_a.ndim)

'''end'''

'''测试tensosrflow的行列优先规则'''
#import tensorflow as tf
#arr_a = tf.zeros([3,2])
#arr_b = np.zeros([3,2])
#print("arr_b:",arr_b)

#with tf.Session() as sess:
#    print("arr_a:",sess.run(arr_a))
#经测得，tensorflow和numpy都是以行为先的。
'''end'''
'''测试数组内的索引判断'''
#arr_a = np.random.normal(size = [2,2])
#print("arr_a:",arr_a)

#import ConvolutionalNetwork as cn
#arr_b = cn.ReLU(arr_a).forward()
#arr_c = cn.ReLU(arr_a).backward()
#print(arr_b)
#print(arr_c)
#arr_a[arr_a < 0]=0.0
#print(arr_a)

#arr_b = np.zeros(arr_a.shape)
#print(arr_b)

#arr_a = np.zeros([3,3])
#arr_a[1][2] = 4
#print(arr_a)
##返回最大的数的索引，不分行和列，是将所有的数展成一维数组的索引。
#print(arr_a.argmax())
#print(arr_a.argmax(axis = 1))

'''end'''

'''test convolutional neural network'''
#import ConvolutionalNetwork as cn

#arr_a = np.random.rand(4,4)
#arr_b = cn.Maxpool(arr_a,np.array([2,2])).backward()
#print(arr_a)
#print(arr_b)

'''end'''

'''测试数组的赋值'''
#arr_a = np.zeros(3)

#arr_b = arr_a
#arr_b[2] = 2

#print(arr_a)
#print(arr_b)
'''输出结果为[0,0,2]  [0,0,2] 可知如果是等号赋值的话，你改变其中一个数组，另外一个数组也会被改变'''

'''end'''

#lbls = np.array([1,2,0,4,3])
#input = np.random.normal(size=[6,5])
#print(input)
##输出lbls列，有5个数，输出5列。对应的行也是一样。
#print(input[:,lbls])
##输出5行，每个行的索引是lbls里面的数。
#print(input[lbls])

#import tensorflow as tf

#input = tf.truncated_normal([4,2])
#sm = tf.nn.softmax(input)

#with tf.Session() as sess:
#    ip = sess.run(input)
#    print(ip)
#    result = sess.run(sm,feed_dict={input : ip})
#    print(result)

#input = tf.truncated_normal([3,4])
#re = tf.argmax(input,1)
#add = tf.multiply(input,0.1)


#with tf.Session() as sess:
#    inp = sess.run(input)
#    print(inp)
#    #result = sess.run(re,feed_dict={input:inp})
#    #print(result)
#    print(sess.run(add,feed_dict={input:inp}))

a = 1
b = "hello"
print("hello %a"%a)#这里其实%s、%d、%f、%c都可以。只是不同的字符对应不同的输出，分别对应str,double,float,character.
print(b+" World")
print(b+ (str)(a))