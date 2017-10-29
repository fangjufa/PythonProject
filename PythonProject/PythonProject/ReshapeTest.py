import numpy as np

import tensorflow as tf

#pic = tf.read_file("00.png")

#print(pic)

x = tf.neg(2)
y = x + x*x

sess = tf.Session()
sess.run(y)

print(y)
