import os
import cv2
import gzip
import numpy as np

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    #if magic != 2051:
    #  raise ValueError('Invalid magic number %d in MNIST image file: %s' %
    #                   (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    #data = data.reshape(num_images, rows, cols, 1)
    return data

def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte.gz')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte.gz')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')

    fimg = open(fname_img, 'rb')

    lbl = extract_images(flbl)
    img = extract_images(fimg)
    img = np.reshape(img,[60000,784])
    lbl = lbl[0:size]
    img = img[0:size]
    ##to get images
    #for i in range(10):
    #    cv2.imwrite("mnistimgs/file%d.jpg"%i,np.reshape(img[i],[28,28,1]))
    return img, lbl