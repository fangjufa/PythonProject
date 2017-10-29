import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
    #neg_images_link = '//image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'   
    neg_images_link = 'lily.txt'
    #neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    neg_image_urls = open(neg_images_link).read()
    pic_num = 1
    
    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            #img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            #resized_image = cv2.resize(img, (100, 100))
            #cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  


'''unpickle cifar images.Read data from zip file.'''
def unpickle(file):
    import pickle as pkl
    with open(file,'rb') as fo:
        dict = pkl.load(fo,encoding = 'bytes')
    return dict
'''call method
dict = unpickle("cifar 10/batches/data_batch_1")
data = dict[b"data"]
labels = dict[b"labels"]
num = data.shape[0]
data = data.reshape(num,3,32,32)
'''