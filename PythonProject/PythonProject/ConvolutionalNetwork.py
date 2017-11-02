import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Convolution(object):
    """convolution layer.default padding is valid,and stride is all 1."""
    def __init__(self,input,conv):

        if input.ndim == 3:
            input = input.reshape([1,input.shape[0],input.shape[1],input.shape[2]])
        elif input.ndim == 2:
            input =input.reshape(1,1,input.shape[0],input.shape[1])
        elif input.ndim > 4:
            raise Exception("Dimension of input array must be less than 4.")

        if conv.ndim == 3:
            conv = conv.reshape([1,conv.shape[0],conv.shape[1],conv.shape[2]])
        elif conv.ndim == 2:
            conv =conv.reshape(1,1,conv.shape[0],conv.shape[1])
        elif conv.ndim > 4:
            raise Exception("Dimension of input array must be less than 4.")

        self.input = input
        self.conv = conv

    def forward(self,backgrad):
        """Convolution operation.Output size is [x.shape[0],conv.shape[0],x.shape[2]-conv.shape[2] + 1,x.shape[3]-conv.shape[3]+1]"""
        x = self.input
        conv = self.conv
        data_num = x.shape[0]
        #output depth.
        depth = conv.shape[0]
        W = x.shape[2]-conv.shape[2] + 1
        H = x.shape[3]-conv.shape[3]+1
        result = np.zeros([data_num,depth,W,H])
        #data number
        for i in range(data_num):
            #extract ith data
            data = x[i,:]
            print("i = ",i)
            for j in range(depth):
                kernel = conv[j,:]
                channel = data.shape[0]
                temp = np.zeros([channel,W,H])
                for k in range(channel):
                    temp[k] = self.conv_simple(data[k],kernel[k])
                result[i][j] = np.sum(temp,axis = 0)
        return result

    def conv_simple(self,x,kernel):
        """A simple one of performing convolution,that x and kernel is a 2D array"""
        if not x.ndim == 2 or not kernel.ndim == 2:
            print("Dimension of x and kernel must be 2.")
            return False
        xw = x.shape[0]
        xh = x.shape[1]

        kw = kernel.shape[0]
        kh = kernel.shape[1]
        #if padding == "valid":
        result = np.zeros([xw-kw + 1,xh-kh +1])
        #else:
        #    result = np.zeros([xw,xh])
    
        for i in range(xw):
            if i + kw > xw:
                break
            for j in range(xh):
                if j+kh > xh:
                    continue
                result[i][j] = np.sum(np.multiply(x[i:kw+i,j:kh+j],kernel))
                

        return result

    def backward(self,backgrad):
        """backpropagation of convolutional operation,it will output two arrays,which is dq/dw and dq/dx"""
        #self.forward()
        x = self.input
        conv = self.conv
        data_num = x.shape[0]
        #output depth.
        depth = conv.shape[0]
        W = x.shape[2]-conv.shape[2] + 1
        H = x.shape[3]-conv.shape[3]+1
        #result = np.zeros([data_num,depth,W,H])
        grad_input = np.zeros(x.shape)
        grad_conv = np.zeros(conv.shape)
        #data number
        for i in range(data_num):
            #extract ith data
            data = x[i,:]
            print("i = ",i)
            for j in range(depth):
                kernel = conv[j,:]
                channel = data.shape[0]
                #temp = np.zeros([channel,W,H])
                for k in range(channel):
                    #[temp[k],self.grad_input[i][k],self.grad_conv[j][k]] = self.conv_simple(data[k],kernel[k])
                    [grad_input[i][k],grad_conv[j][k]] = self.conv_simple(data[k],kernel[k],backgrad[i][j])
                #result[i][j] = np.sum(temp,axis = 0)

        return [grad_input,grad_conv]

    def backward_simple(self,backgrad):
        if not x.ndim == 2 or not kernel.ndim == 2:
            print("Dimension of x and kernel must be 2.")
            return False
        xw = x.shape[0]
        xh = x.shape[1]

        kw = kernel.shape[0]
        kh = kernel.shape[1]

        grad_kernel = np.zeros(kernel.shape)
        grad_x = np.zeros(x.shape)
        #if padding == "valid":
        #result = np.zeros([xw-kw + 1,xh-kh +1])
        #else:
        #    result = np.zeros([xw,xh])
    
        for i in range(xw):
            if i + kw > xw:
                break
            for j in range(xh):
                if j+kh > xh:
                    continue
                grad_kernel += x[i:kw + i,j:kh + j]*backgrad[i][j]
                grad_x[i:kw+i,j:kh+j] += kernel*backgrad[i][j]
                #result[i][j] = np.sum(np.multiply(x[i:kw+i,j:kh+j],kernel))
        return [grad_x,grad_kernel]


class Maxpool(object):
    def __init__(self, input,kernel):
        #if not input.ndim == 4 or not kernel.ndim == 1:
        #    print("Dimensions of x must be 4,so as kernel must be 1.")
        #    return None
        self.input = input
        self.kernel = kernel
        x = self.input
        if x.ndim == 3:
            x = x.reshape([1,x.shape[0],x.shape[1],x.shape[2]])
        elif x.ndim == 2:
            x =x.reshape(1,1,x.shape[0],x.shape[1])
        elif x.ndim > 4:
            raise Exception("Dimension of input array must be less than 4.")
        
        xw = x.shape[2]
        xh = x.shape[3]
        kw = kernel[0]
        kh = kernel[1]
        self.input = x
        if not xw % kw == 0 or not xh%kh == 0:
            raise Exception("width of input must be times of width of kernel.So as height of input and height of kernel.")
        
    def maxpool_simple(self,x,kernel):
        """Simple maxpool,that dimensions of x and kernel is just 2."""
        #if not x.ndim == 2 or not kernel.ndim == 1:
        #    print("Simple max_pool,dimensions of x and kernel must be 2.")
        #    return False
        xw = x.shape[0]
        xh = x.shape[1]
        kw = kernel[0]
        kh = kernel[1]
        w = int(xw/kw)
        h = int(xh/kh)
        result = np.zeros([w,h])

        for i in range(w):
            for j in range(h):
                result[i][j] = np.max(x[i*kw:(i+1)*kw,j*kh:(j+1)*kh])
                #dx[i*kw:(i+1)*kw,j*kh:(j+1)*kh] += dre[i][j] * 
                
        return result

    def backprop_simple(self,x,kernel,backgrad):
        xw = x.shape[0]
        xh = x.shape[1]
        kw = kernel[0]
        kh = kernel[1]
        w = int(xw/kw)
        h = int(xh/kh)

        result = np.zeros([xw,xh])

        for i in range(w):
            for j in range(h):
                temp = np.zeros(x[i*kw:(i+1)*kw,j*kh:(j+1)*kh].shape)
                maxindex = np.argmax(x[i*kw:(i+1)*kw,j*kh:(j+1)*kh])
                
                temp[int(maxindex/kw)][maxindex%kh] = 1
                result[i*kw:(i+1)*kw,j*kh:(j+1)*kh] += backgrad[i][j] * temp
        return result

    def forward(self):
        """Max pool operation.It would change the shape of input.The output size is [x.shape[0],x.shape[1],x.shape[2]/kernel[0],x.shape[3]/kernel[1]]"""
        
        x = self.input
        kernel = self.kernel
        data_num = x.shape[0]
        channel = x.shape[1]
        xw = x.shape[2]
        xh = x.shape[3]
        kw = kernel[0]
        kh = kernel[1]

        if not xw%kw == 0 or not xh%kh== 0:
            print("Width of X must be integral times of kernel width,and so as height.")
            return None
        result = np.zeros([data_num,channel,int(xw/kw),int(xh/kh)])
        for i in range(data_num):
            data = x[i,:]
            for j in range(channel):
                result[i][j] = self.maxpool_simple(data[j],kernel)
        return result

    def backward(self,backgrad):
        """maxpool backpropagation."""
        x = self.input
        kernel = self.kernel

        data_num = x.shape[0]
        channel = x.shape[1]
        xw = x.shape[2]
        xh = x.shape[3]
        kw = kernel[0]
        kh = kernel[1]
        output = np.zeros(x.shape)
        for i in range(data_num):
            data = x[i,:]
            for j in range(channel):
                output[i][j] = self.backprop_simple(data[j],kernel,backgrad[i][j])
        return output


class ReLU(object):
    def __init__(self, input):
        self.input = input
        self.output = np.copy(input)

    def forward(self):
        """ReLU operation.This is imple,that if the element is less equal to 0,then set it to 0."""
        self.output[self.output < 0] = 0
        return self.output

    def backward(self,backgrad):
        """ReLU backpropagation.If x > 0,then f(x) = x,f'(x) = 1.otherwise f'(x) = 0."""
        back = np.copy(self.output)
        back[back > 0] = 1
        back[back < 0] = 0
        back = back*backgrad
        return back
    
#arr_a = np.array([[[[1,2,4],[2,4,3],[5,5,2]],[[1,1,2],[2,5,4],[3,2,6]],[[6,5,2],[1,2,2],[3,2,4]]]])
##arr_b = np.array([[2,1],[2,4]])
###result = Convolution(arr_a,arr_b).backward_simple(arr_a,arr_b)
##result = Convolution(arr_a,arr_b).backward()
##print(result[0])
##print(result[1])
#arr_b = np.array([[[[3,2],[1,1]],[[2,4],[5,6]],[[8,2],[3,6]]]])

#arr1 = np.array([[1,2,4],[2,4,3],[5,5,2]])
#kernel1 = np.array([[3,2],[1,1]])

#arr2 = np.array([[1,1,2],[2,5,4],[3,2,6]])
#kernel2 = np.array([[2,4],[5,6]])

#arr3 = np.array([[6,5,2],[1,2,2],[3,2,4]])
#kernel3 = np.array([[8,2],[3,6]])

#result1 = Convolution(arr_a,arr_b).conv_simple(arr1,kernel1)
#result2 = Convolution(arr_a,arr_b).conv_simple(arr2,kernel2)
#result3 = Convolution(arr_a,arr_b).conv_simple(arr3,kernel3)

##final = np.array([result1,result2,result3])
#print("result1",result1)
#print("result2",result2)
#print("result3",result3)

##result = Convolution(arr_a,arr_b).backward()
##print("numpy grad_c:",result[0])
##print("numpy grad_d:",result[1])

#result = Convolution(arr_a,arr_b).forward()
#print("numpy Convolution:",result)

#import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

##arr_c = tf.Variable([[[[1,2,4],[2,4,3],[5,5,2]],[[1,1,2],[2,5,4],[3,2,6]]]],dtype = tf.float32)#,[[6,5,2],[1,2,2],[3,2,4]]]],dtype = tf.float32)
##arr_d = tf.Variable([[[[3,2],[1,1]],[[2,4],[5,6]]]],dtype = tf.float32)#,[[8,2],[3,6]]]],dtype = tf.float32)
##reshape_c = tf.reshape(arr_c,[1,3,3,2])
##reshape_d = tf.reshape(arr_d,[2,2,2,1])
##conv = tf.nn.conv2d(reshape_c,reshape_d,[1,1,1,1],"VALID")

##grad_c = tf.gradients(conv,arr_c)
##grad_d = tf.gradients(conv,arr_d)

#arr_c = tf.Variable([[[[1,2,4],[2,4,3],[5,5,2]],[[1,1,2],[2,5,4],[3,2,6]],[[6,5,2],[1,2,2],[3,2,4]]]],dtype = tf.float32)
##arr_d = tf.Variable([[[[3,2],[1,1]],[[2,4],[5,6]],[[8,2],[3,6]]]],dtype = tf.float32)
#arr_d = tf.Variable([3,2,8,2,4,2,1,5,3,1,6,6],dtype = tf.float32)
#reshape_c = tf.reshape(arr_c,[1,3,3,3])
#reshape_d = tf.reshape(arr_d,[2,2,3,1])
#conv = tf.nn.conv2d(reshape_c,reshape_d,[1,1,1,1],"VALID")

##def weight_variable(shape):
##    initial = tf.truncated_normal(shape,stddev = 0.1)
##    return tf.Variable(initial)

##w = weight_variable([1,3,3,3])
##conv = weight_variable([1,2,2,3])

##tf.nn.conv2d(w,conv,[]
#initialize = tf.global_variables_initializer()

#with tf.Session() as sess:
#    sess.run(initialize)
#    conv_result = sess.run(conv)
#    print("Tensorflow result:",conv_result)
#    print("shape_c:",sess.run(reshape_c))
#    print("shape_d:",sess.run(reshape_d))
    #result_c = sess.run(grad_c)
    #result_d = sess.run(grad_d)

    #print("Tensorflow grad_c:",result_c)
    #print("Tensorflow grad_d:",result_d)

input = np.random.normal(size = [1,1,4,4])
W = np.random.normal(size = [1,1,2,2])

W1 = Convolution(input,W).forward()

W2 = W1 * 4
W3 = np.multiply( W2,W2)

dW2 = 2*W2
dW1 = 4*dW2

dW = Convolution(input,W).backward(dW1)

print("Numpy:",dW)


import tensorflow as tf
tfInput = tf.Variable(input)
tfW = tf.Variable(W)
tfW = tf.reshape(tfW,shape = [1,4,4,1])
tfRe = tf.nn.conv2d(tfW,[1,2,2,1],[1,2,2,1],padding="VALID")
tfResult = (tfRe*4)**2

grad = tf.gradients(tfResult,tfW)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    g_re = sess.run(grad)
    print("Tensorflow:",g_re)