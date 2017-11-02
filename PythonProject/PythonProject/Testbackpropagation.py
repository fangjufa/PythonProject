import numpy as np
from random import seed
from random import random
from math import exp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def var_initial(shape):
    return np.random.normal(size = shape)

a = var_initial([4,5])
x = var_initial([5,2])

ax = np.dot(a,x)
eax = np.exp(ax)

b = var_initial([2])

#re = eax + b
re = np.zeros([4,2])

db = np.zeros(b.shape)
deax = np.zeros(eax.shape)

for i in range(4):
    re[i] = eax[i] + b
    db = np.add(db,np.ones(b.shape))
    deax[i] = np.add(deax[i],np.ones(b.shape))

#print(db)
#print(deax)

dax = np.exp(ax) * deax

x = x.transpose()
da = np.zeros([4,5])
dx = np.zeros([2,5])
for i in range(4):
    for j in range(2):
        #dot = np.dot(a[i],x[j])
        da[i] += x[j] * dax[i][j]
        dx[j] += a[i] * dax[i][j]
print("numpy a",da)
print("numpy x",dx.transpose())
print("numpy b",db)
    
import tensorflow as tf

tfa = tf.Variable(a)
tfx = tf.Variable(x.transpose())
tfb = tf.Variable(b)

re = tf.exp(tf.matmul(tfa,tfx)) + tfb

g_tfa = tf.gradients(re,tfa)
g_tfx = tf.gradients(re,tfx)
g_tfb = tf.gradients(re,tfb)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(re)
    resulta = sess.run(g_tfa)
    resultx = sess.run(g_tfx)
    resultb = sess.run(g_tfb)

    print("tensorflow a:",resulta)
    print("tensorflow x:",resultx)
    print("tensorflow b:",resultb)

    

#import tensorflow as tf
#def var02_initial(shape):
#    initial = tf.truncated_normal(shape,stddev = 0.1)
#    return tf.Variable(initial)
#    #tensorflow运算输入一定要是它的tensor量，numpy的数组也不能计算，会报类型错误。
#y = var02_initial([4,2])
#z = var02_initial([2])

#l = y + z
#grad_z = tf.gradients(l,z)
#grad_y = tf.gradients(l,y)

#with tf.Session() as sess:
#    gradz = sess.run(grad_z)
#    grady = sess.run(grad_y)
#    #该输出结果为[4,4]
#    print(gradz)
#    #该输出结果为[[1,1],[1,1],[1,1],[1,1]]
#    print(grady)


#def initialize_network(n_inputs, n_hidden, n_outputs):
#	network = list()
#	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
#	network.append(hidden_layer)
#	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
#	network.append(output_layer)
#	return network

##	activation = sum(weight_i * input_i) + bias
#def activate(weights, inputs):
#	activation = weights[-1]
#	for i in range(len(weights)-1):
#		activation += weights[i] * inputs[i]
#	return activation

##output = 1 / (1 + e^(-activation))
#def transfer(activation):
#	return 1.0 / (1.0 + exp(-activation))

## Forward propagate input to a network output
#def forward_propagate(network, row):
#	inputs = row
#	for layer in network:
#        new_inputs = []
#        activation = activate(layer, inputs)
#        output = transfer(activation)
#        new_inputs.append(output)
#		inputs = new_inputs
#	return inputs

## test forward propagation
##network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
##		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#network = [[0.1, 0.8, 0.7],[0.2, 0.4, 0.6]]
#row = [1, 0, 0.5]
#output = forward_propagate(network, row)
#print(output)

## Calculate the derivative of an neuron output
#def transfer_derivative(output):
#	return output * (1.0 - output)

## Backpropagate error and store in neurons
#def backward_propagate_error(network, expected):
#	for i in reversed(range(len(network))):
#		layer = network[i]
#		errors = list()
#		if i != len(network)-1:
#			for j in range(len(layer)):
#				error = 0.0
#				for neuron in network[i + 1]:
#					error += (neuron['weights'][j] * neuron['delta'])
#				errors.append(error)
#		else:
#			for j in range(len(layer)):
#				neuron = layer[j]
#				errors.append(expected[j] - neuron['output'])
#		for j in range(len(layer)):
#			neuron = layer[j]
#			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

## test backpropagation of error
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)