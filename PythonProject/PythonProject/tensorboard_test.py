﻿#from __future__ import absolute_import  
#from __future__ import division  
#from __future__ import print_function  
  
#import argparse  
#import sys  
  
#import tensorflow as tf  
  
#from tensorflow.examples.tutorials.mnist import input_data  
  
#FLAGS = None  
  
  
#def train():  
#  # Import data  
#  mnist = input_data.read_data_sets("mnist_train/",#FLAGS.data_dir,  
#                                    one_hot=True)
#                                    #fake_data=FLAGS.fake_data)  

#  sess = tf.InteractiveSession()  
#  # Create a multilayer model.  
  
#  # Input placeholders  
#  with tf.name_scope('input'):  
#    x = tf.placeholder(tf.float32, [None, 784], name='x-input')  
#    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')  
  
#  with tf.name_scope('input_reshape'):  
#    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])  
#    tf.summary.image('input', image_shaped_input, 10)  
  
#  # We can't initialize these variables to 0 - the network will get stuck.  
#  def weight_variable(shape):  
#    """Create a weight variable with appropriate initialization."""  
#    initial = tf.truncated_normal(shape, stddev=0.1)  
#    return tf.Variable(initial)  
  
#  def bias_variable(shape):  
#    """Create a bias variable with appropriate initialization."""  
#    initial = tf.constant(0.1, shape=shape)  
#    return tf.Variable(initial)  
  
#  def variable_summaries(var):  
#    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""  
#    with tf.name_scope('summaries'):  
#      mean = tf.reduce_mean(var)  
#      tf.summary.scalar('mean', mean)  
#      with tf.name_scope('stddev'):  
#        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  
#      tf.summary.scalar('stddev', stddev)  
#      tf.summary.scalar('max', tf.reduce_max(var))  
#      tf.summary.scalar('min', tf.reduce_min(var))  
#      tf.summary.histogram('histogram', var)  
  
#  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):  
#    """Reusable code for making a simple neural net layer.  
#    It does a matrix multiply, bias add, and then uses relu to nonlinearize.  
#    It also sets up name scoping so that the resultant graph is easy to read,  
#    and adds a number of summary ops.  
#    """  
#    # Adding a name scope ensures logical grouping of the layers in the graph.  
#    with tf.name_scope(layer_name):  
#      # This Variable will hold the state of the weights for the layer  
#      with tf.name_scope('weights'):  
#        weights = weight_variable([input_dim, output_dim])  
#        variable_summaries(weights)  
#      with tf.name_scope('biases'):  
#        biases = bias_variable([output_dim])  
#        variable_summaries(biases)  
#      with tf.name_scope('Wx_plus_b'):  
#        preactivate = tf.matmul(input_tensor, weights) + biases  
#        tf.summary.histogram('pre_activations', preactivate)  
#      activations = act(preactivate, name='activation')  
#      tf.summary.histogram('activations', activations)  
#      return activations  
  
#  hidden1 = nn_layer(x, 784, 500, 'layer1')  
  
#  with tf.name_scope('dropout'):  
#    keep_prob = tf.placeholder(tf.float32)  
#    tf.summary.scalar('dropout_keep_probability', keep_prob)  
#    dropped = tf.nn.dropout(hidden1, keep_prob)  
  
#  # Do not apply softmax activation yet, see below.  
#  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)  
  
#  with tf.name_scope('cross_entropy'):  
#    # The raw formulation of cross-entropy,  
#    #  
#    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),  
#    #                               reduction_indices=[1]))  
#    #  
#    # can be numerically unstable.  
#    #  
#    # So here we use tf.nn.softmax_cross_entropy_with_logits on the  
#    # raw outputs of the nn_layer above, and then average across  
#    # the batch.  
#    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)  
#    with tf.name_scope('total'):  
#      cross_entropy = tf.reduce_mean(diff)  
#  tf.summary.scalar('cross_entropy', cross_entropy)  
  
#  with tf.name_scope('train'):  
#    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(  
#        cross_entropy)  
  
#  with tf.name_scope('accuracy'):  
#    with tf.name_scope('correct_prediction'):  
#      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
#    with tf.name_scope('accuracy'):  
#      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
#  tf.summary.scalar('accuracy', accuracy)  
  
#  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)  
#  merged = tf.summary.merge_all()  
#  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)  
#  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')  
#  tf.global_variables_initializer().run()  
  
#  # Train the model, and also write summaries.  
#  # Every 10th step, measure test-set accuracy, and write test summaries  
#  # All other steps, run train_step on training data, & add training summaries  
  
#  def feed_dict(train):  
#    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""  
#    if train or FLAGS.fake_data:  
#      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)  
#      k = FLAGS.dropout  
#    else:  
#      xs, ys = mnist.test.images, mnist.test.labels  
#      k = 1.0  
#    return {x: xs, y_: ys, keep_prob: k}  
  
#  for i in range(FLAGS.max_steps):  
#    if i % 10 == 0:  # Record summaries and test-set accuracy  
#      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))  
#      test_writer.add_summary(summary, i)  
#      print('Accuracy at step %s: %s' % (i, acc))  
#    else:  # Record train set summaries, and train  
#      if i % 100 == 99:  # Record execution stats  
#        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  
#        run_metadata = tf.RunMetadata()  
#        summary, _ = sess.run([merged, train_step],  
#                              feed_dict=feed_dict(True),  
#                              options=run_options,  
#                              run_metadata=run_metadata)  
#        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)  
#        train_writer.add_summary(summary, i)  
#        print('Adding run metadata for', i)  
#      else:  # Record a summary  
#        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))  
#        train_writer.add_summary(summary, i)  
#  train_writer.close()  
#  test_writer.close()  
  
  
#def main(_):  
#  if tf.gfile.Exists(FLAGS.log_dir):  
#    tf.gfile.DeleteRecursively(FLAGS.log_dir)  
#  tf.gfile.MakeDirs(FLAGS.log_dir)  
#  train()  
  
  
#if __name__ == '__main__':  
#  parser = argparse.ArgumentParser()  
#  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,  
#                      default=False,  
#                      help='If true, uses fake data for unit testing.')  
#  parser.add_argument('--max_steps', type=int, default=500,  
#                      help='Number of steps to run trainer.')  
#  parser.add_argument('--learning_rate', type=float, default=0.001,  
#                      help='Initial learning rate')  
#  parser.add_argument('--dropout', type=float, default=0.9,  
#                      help='Keep probability for training dropout.')  
#  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',  
#                      help='Directory for storing input data')  
#  parser.add_argument('--log_dir', type=str, default='C:/tmp/tensorflow/mnist/logs/mnist_with_summaries',  
#                      help='Summaries log directory')  
#  FLAGS, unparsed = parser.parse_known_args()  
#  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  


#import tensorflow as tf
#with tf.variable_scope("foo"):
#    a = tf.get_variable("bar", [1])
#    print (a.name)

#with tf.variable_scope("bar"):
#    b = tf.get_variable("bar", [1])
#    print (b.name)

#with tf.name_scope("a"):
#    a = tf.Variable([1])
#    print (a.name)

#    a = tf.get_variable("b", [1])
#    print (a.name)

#with tf.name_scope("input1"):
#    input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
#with tf.name_scope("input2"):
#    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
#output = tf.add_n([input1, input2], name="add")

#writer = tf.summary.FileWriter("E:\logs", tf.get_default_graph())
#writer.close()

#import tensorflow as tf
#C_1 = tf.constant(5.0)
#C_2 = tf.constant(1.0)
#C_3 = tf.constant(2.0)

#golden_ratio = (tf.sqrt(C_1) + C_2)/C_3

#with tf.Session() as sess:
#    writer = tf.summary.FileWriter('logs', sess.graph)
#    print( sess.run(golden_ratio))
#    writer.close()


'''------------------------'''

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#A simple MNIST classifier which displays summaries in TensorBoard.
# This is an unimpressive MNIST model, but it is a good example of using
#tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
#naming summary tags so that they are grouped meaningfully in TensorBoard.
#It demonstrates the functionality of every TensorBoard dashboard.
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')


def train():
  # Import data
  mnist = input_data.read_data_sets("mnist_train/",#FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.image_summary('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()


if __name__ == '__main__':
  tf.app.run()