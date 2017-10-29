# Typical setup to include TensorFlow.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

#from PIL import Image

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
#创建一个队列(Queue)，将所有的文件名字加入到这个队列当中
filename_queue = tf.train.string_input_producer(
    #["http://visboo.com/img/new/352597.jpg"])
    #tf.train.match_filenames_once("images/*.jpg")) #查找到所有匹配字符串的图片名字
    ["images/curry.jpg","images/curry02.jpg"]) #也可以像这样，将所有的图片名字列出来

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
#图片读取器
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
#read函数将文件名队列作为参数，输出一对Key Value
#Key是文件名字，Value是文件内容。
key,image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this wi ll turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)


# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    
    #print("file list:",sess.run(file_list))

    # Get an image tensor and print its value.
    #length of your filename list
    print(sess.run(key))
    image01 = image.eval() #here is your image Tensor :) 

    print(image01.shape)
    
    #Image.fromarray(np.asarray(image01)).show()
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)