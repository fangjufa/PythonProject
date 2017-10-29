import tensorflow as tf
#一个Queue,用来保存文件名字.对此Queue,只读取,不dequeue
filename_queue = tf.train.string_input_producer(["1.txt", "WebGL.docx"])

#用来从文件中读取数据, LineReader,每次读一行
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  #在调用run或eval执行读取之前，必须
  #用tf.train.start_queue_runners来填充队列
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(10):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])
    print(example, label)
  coord.request_stop()
  coord.join(threads)