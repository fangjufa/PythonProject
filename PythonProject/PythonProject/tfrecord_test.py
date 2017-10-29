import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter("test.tfrecord")

for i in range(0,2):
    a = 0.68 + i
    b = [1 + i,2 + i]
    c = "Hello:" + str(i)
    c_raw = c

    example = tf.train.Example(
        feature = {'a':tf.train.Feature(float_list = tf.train.FloatList(value=[a])),
                       'b':tf.train.Feature(int64_list = tf.train.Int64List(value = b)) })

    serialized = example.SerializeToString()

    writer.write(serialized)

print("done")
'''c':tf.train.Feature(bytes_list = tf.train.BytesList(value = [c_raw]))'''