# Typical setup to include TensorFlow.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#创建一个队列(Queue)，将所有的文件名字加入到这个队列当中
filename_queue = tf.train.string_input_producer(
    #["http://visboo.com/img/new/352597.jpg"])
    #tf.train.match_filenames_once("images/*.jpg")) #查找到所有匹配字符串的图片名字
    ["images/curry.jpg"]) #也可以像这样，将所有的图片名字列出来

#图片读取器
image_reader = tf.WholeFileReader()

#read函数将文件名队列作为参数，输出一对Key Value
#Key是文件名字，Value是文件内容。
key,image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this wi ll turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)

#image_mid = tf.reshape(image,[1,tf.shape(image)[0],tf.shape(image)[1],tf.shape(image)[2]])
image_input = tf.cast([image],tf.float32)

#conv_filter = tf.constant([[[[]]])
#image_const = tf.constant(image_input,tf.float32)
image_pool = tf.nn.max_pool(image_input,[1,5,5,1],[1,5,5,1],'SAME')

image_show = tf.cast(tf.reshape(image_pool,[tf.shape(image_pool)[1],tf.shape(image_pool)[2],tf.shape(image_pool)[3]]),tf.uint8)
# Start a new session to show example output.
with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    img = sess.run(image_input)
    #print("image_original",img.shape())
    
    #Image.fromarray(np.asarray(image01)).show()
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    image_tensor = sess.run(image_show)

   
    print("image_tensor",image_tensor.shape())

    plt.figure()
    #image_pool = sess.run(tf.nn.max_pool(image_tensor,[1,2,2,1],[1,2,2,1],'VALID'))
    #print(image_pool.shape)

    #image_show = sess.run()

    plt.imshow(image_tensor)
    plt.show()