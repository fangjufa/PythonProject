#Use linear classify to train mnist images and do some tests.
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

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

def predict(W,b,img):
    max_score = -9999.0
    max_index = -1
    #calculate scores
    for j in range(10):
        score = np.dot(W[j],img) + b[j]
        if score > max_score:
            max_score = score
            max_index = j
    return max_index    
    
#read textures
cur_time = time.clock()
train_lbls,train_imgs,test_lbls,test_imgs = read_mnist()
print("Done for read mnist,cost %f seconds."%(time.clock() - cur_time))

#there are 10 classes,each respect to 0～9
W = np.random.rand(10,784) - 0.5
b = np.random.rand(10) - 0.5

#original learning rate,it should be small one,
#because it's very easy to overflow when doing exp operation.
original_lr = 0.1
learning_rate = original_lr
scores = np.zeros(10)
total_loss = 9999

#draw loss plots
plt.ion() #important,turn interactive mode on.
figure = plt.figure()

axes = figure.add_subplot(111)
axes.set_xlim([0,51])
axes.set_ylim([0,0.2])
x_data = np.linspace(0,50)
y_data = np.zeros(50)
line,= axes.plot(0, 0, 'r')

for count in range(50):
    losses = np.zeros(7000)
    gradW = np.zeros([10,784])
    gradb = np.zeros(10)

    #testgW = np.zeros([10,784])
    #testgb = np.zeros(10)
    for i in range(7000):
           
        #label of i
        li = train_lbls[i]
        import lossfunction as lf
        gw,gb,losses[i] = lf.LossFuntion().SoftMax(W,b,train_imgs[i],li)
        gradW = np.add(gradW,gw)
        gradb = np.add(gradb, gb)

    W += -learning_rate*gradW/7000
    b += -learning_rate*gradb/7000
    losses /= 7000
    total_loss = 10*np.sum(losses)/7000
    print("Now loss is %f,and cost %f seconds time.learning rate is %f."%( total_loss,time.clock() - cur_time,learning_rate))

    count += 1
    learning_rate = original_lr/(1.0 + count/10)

    #draw loss plots
    if total_loss == np.inf or total_loss == np.nan:
        total_loss = 0

    line.set_xdata(np.append(line.get_xdata(), count))
    line.set_ydata(np.append(line.get_ydata(), total_loss))

    #y_data[count] = total_loss
    #line.set_ydata(y_data)
    figure.canvas.draw()

plt.show()
correct_count = 0
for i in range(1000):
    pred_lable = predict(W,b,test_imgs[i])
    #print("pred_lable:%d ,true label:%d"%(pred_lable,test_lbls[i]))
    if pred_lable == test_lbls[i]:
        correct_count += 1

print("The predict accuracy is %f "%(correct_count/1000))