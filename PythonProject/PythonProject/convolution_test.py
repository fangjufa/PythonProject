import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
#image_raw is a three dimension tensor,which dimension represents to height,width,channel of this image.
image_raw = tf.image.decode_jpeg(image_file)
image = tf.reshape(image_raw,[-1,160,219,3])#image_raw.shape[0],image_raw.shape[1],image_raw.shape[2]])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess = sess,coord = coord)

image_tensor = sess.run(image)
print(image_tensor.shape)

#image_show = sess.run(image_final)
#image_tensor = tf.cast(tf.reshape(image,[1,image_tensor[0],image_tensor[1],image_tensor[2]]),tf.float32)

#image_show = sess.run(image_final,feed_dict = {image_holder:image_tensor})
#image_show = sess.run(image_conv)
#image_show02 = tf.cast(tf.reshape(image_show,[image_show.shape[1],image_show.shape[2],image_show.shape[3]]),tf.uint8)
#print(image_show02._dtype)

coord.request_stop()
coord.join(threads)

plt.figure()
plt.imshow(image_tensor)
plt.show()