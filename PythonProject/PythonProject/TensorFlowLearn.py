import tensorflow as tf
#定义一个两行三列的矩阵
matrix1 = tf.constant([[2,1,3],[5,2,4]])

#定义另一个三行两列的矩阵
matrix2 = tf.constant([[1,5],[1,3],[2,1]])

#求两个矩阵的乘法值
product = tf.matmul(matrix1,matrix2)

#定义一个标量的变量
state = tf.Variable(2,None)

#计算一个矩阵与一个标量的乘积
mult = tf.multiply(matrix1,state)

mat_grad = tf.gradients(mult,matrix1)
sta_grad = tf.gradients(mult,state)


#因为上面有变量定义，所以需要初始化变量
initvar = tf.global_variables_initializer()

#启动会话,注意这个Session函数的大小写
with tf.Session() as sess:

    #设置使用特定的某个设备来执行，/gpu:0代表由第一个gpu来执行这个会话，/gpu:1等以此类推
    #但当我输入10000时，并没有报错，可能是当找不到指定的gpu时，它会自动使用默认的设备计算。
    #with tf.device("/gpu:0"):

    #先运行初始化变量op
    sess.run(initvar)
    result = sess.run(product)
    #当你需要获取多个op计算的结果时，调用多次run，传入你想获取的变量。
    result2 = sess.run(mult)
    print(result)
    print("multiply:",result2)
    #你也可以只调用一次run函数，获取多个值，如下所示
    #result3 = sess.run([product,mult])
    #print("result3:",result3[0],result3[1])

    grad = sess.run([mat_grad,sta_grad])
    print("gradient of mat:",grad[0])
    print("gradient of state:",grad[1])

#sess.close()

