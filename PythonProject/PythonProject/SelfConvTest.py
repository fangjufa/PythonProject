import ConvolutionalNetwork as cn
import numpy as np
import cv2
import ConvolutionalNetwork as cn
import lossfunction as lf
def read_mnist():
    train_labels = np.zeros(5000,dtype = np.int32)
    #train_imgs = np.zeros([7000,784],dtype = np.uint8)
    train_imgs = np.zeros([5000,1,28,28],dtype = np.uint8)
    test_labels = np.zeros(800,dtype = np.int32)
    #test_imgs = np.zeros([1000,784],dtype = np.uint8)
    test_imgs = np.zeros([800,1,28,28],dtype = np.uint8)

    file = "mnist_train/"
    for i in range(10):
        imgi = file + "train%d"%i
        train_labels[500*i:500*(i+1)] = i
        test_labels[80*i:80*(i+1)] = i
        for j in range(580):
            num_img = cv2.imread(imgi + "/train%d_%d.jpg"%(i,j),cv2.IMREAD_GRAYSCALE)
            #580 images,500 for train,and 80 for test
            if j >= 500:
                test_imgs[i*80 + j-500] = np.reshape(num_img,[1,28,28])
            else:
                train_imgs[i*500+j] = np.reshape(num_img,[1,28,28])

    return [train_labels,train_imgs,test_labels,test_imgs]

def weight_initial(shape):
    return np.random.normal(size = shape)

def calculate_loss(W1,W2,W3,b):
    global train_imgs,train_lbls

    grad_w1 = np.zeros(W1.shape)
    grad_w2 = np.zeros(W2.shape)
    grad_w3 = np.zeros(W3.shape)
    grad_input = np.zeros([5000,1024])
    grad_b = np.zeros(b.size)
    #first convolution
    conv1 = cn.Convolution(train_imgs,W1).forward()

    ReLU1 = cn.ReLU(conv1).forward()

    max_pool1 = cn.Maxpool(ReLU1,np.array([2,2])).forward()

    '''second convolution'''
    conv2 = cn.Convolution(max_pool1,W2).forward()

    ReLU2 = cn.ReLU(conv2).forward()

    max_pool2 = cn.Maxpool(ReLU2,np.array([2,2])).forward()

    '''fully connected operation'''
    input = max_pool2.reshape(5000,1024)

    loss = np.zeros(5000)
    for i in range(5000):
        [grad_input[i], gw3,gb, loss[i]] = lf.LossFuntion().WW(W3,b,input[i],train_lbls[i])
        grad_w3 = np.add(grad_w3,gw3)
        grad_b = np.add(grad_b, gb)


    total_loss = np.sum(loss)/5000
    print("Total loss is ",total_loss)
    return []


'''
train_lbls [7000],train_imgs [7000,1,28,28],test_lbls [1000],test_imgs [1000,1,28,28]
'''
[train_lbls,train_imgs,test_lbls,test_imgs] = read_mnist()

'''
first convolution
'''
W1 = weight_initial([32,1,5,5])
conv1 = cn.Convolution(train_imgs,W1).forward()

ReLU1 = cn.ReLU(conv1).forward()

max_pool1 = cn.Maxpool(ReLU1,np.array([2,2])).forward()

'''second convolution'''
W2 = weight_initial([64,32,5,5])
conv2 = cn.Convolution(max_pool1,W2).forward()

ReLU2 = cn.ReLU(conv2).forward()

max_pool2 = cn.Maxpool(ReLU2,np.array([2,2])).forward()

'''fully connected operation'''
W3 = weight_initial([10,1024])
input = max_pool2.reshape(5000,1024)
[grad_w,grad_input,loss] = lf.LossFuntion().WW(W3,input,train_lbls)
learning_rate = 0.01
