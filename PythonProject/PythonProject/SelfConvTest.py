import ConvolutionalNetwork as cn
import numpy as np
import cv2
import ConvolutionalNetwork as cn
import lossfunction as lf
def read_mnist():
    train_labels = np.zeros(7000,dtype = np.int32)
    train_imgs = np.zeros([7000,784],dtype = np.uint8)
    test_labels = np.zeros(1000,dtype = np.int32)
    test_imgs = np.zeros([1000,784],dtype = np.uint8)

    file = "mnist_train/"
    for i in range(10):
        imgi = file + "train%d"%i
        train_labels[700*i:700*(i+1)] = i
        test_labels[100*i:100*(i+1)] = i
        for j in range(800):
            num_img = cv2.imread(imgi + "/train%d_%d.jpg"%(i,j),cv2.IMREAD_GRAYSCALE)
            #800 images,700 for train,and 100 for test
            if j >= 700:
                test_imgs[i*100 + j-700] = np.reshape(num_img,[784])
            else:
                train_imgs[i*700+j] = np.reshape(num_img,[784])

    return [train_labels,train_imgs,test_labels,test_imgs]

def weight_initial(shape):
    return np.random.normal(size = shape)

'''
train_lbls [7000],train_imgs [7000,784],test_lbls [1000],test_imgs [1000,784]
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



