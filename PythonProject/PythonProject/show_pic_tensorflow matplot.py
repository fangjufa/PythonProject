import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import numpy as np

#创建一个队列(Queue)，将所有的文件名字加入到这个队列当中
filename_queue = tf.train.string_input_producer(
    #tf.train.match_filenames_once("images/*.jpg")) #查找到所有匹配字符串的图片名字
    ["curry.jpg"]) #也可以像这样，将所有的图片名字列出来

#图片读取器
image_reader = tf.WholeFileReader()

#read函数将文件名队列作为参数，输出一对Key Value
#Key是文件名字，Value是文件内容。
key,image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)
image_shape = tf.reshape(image, [1,tf.shape(image)[0],tf.shape(image)[1],tf.shape(image)[2]])
image_float = tf.cast(image_shape,tf.float32)
#image_conv = tf.nn.conv2d(image_float,
image_pool = tf.nn.max_pool(image_float,[1,5,5,1],[1,3,3,3],'VALID')
image_final = tf.cast(tf.reshape(image_pool,[tf.shape(image_pool)[1],tf.shape(image_pool)[2],tf.shape(image_pool)[3]]),tf.uint8)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Get an image tensor and print its value.
    #length of your filename list
    
    image_tensor=sess.run(image_final)#tf.nn.max_pool(image_shape,[1,2,2,1],[1,2,2,1],padding = 'VALID'))

    #print("imageshape:",image_shape)
    #image_tensor = image.eval() #here is your image Tensor :) 

    #Image.fromarray(np.asarray(image01)).show()
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    #print("image_tensor:",image_tensor.shape[2])
    fig = plt.figure()

    #plt.imshow(sess.run(image))

    fig.add_subplot(1,2,1)
    plt.title("original")
    plt.imshow(sess.run(image))
    fig.add_subplot(1,2,2)
    plt.title("pool over")
    plt.imshow(image_tensor)
    plt.show()
