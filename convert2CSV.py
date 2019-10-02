from mlxtend.data import loadlocal_mnist
from numpy import savetxt

train = 0
if train == 1:
    X, y = loadlocal_mnist( images_path='data/train-images.idx3-ubyte', labels_path='data/train-labels.idx1-ubyte')
    savetxt(fname='data/train_images.csv', X=X, delimiter=',', fmt='%d')
    savetxt(fname='data/train_labels.csv', X=y, delimiter=',', fmt='%d')
else:
    X, y = loadlocal_mnist( images_path='data/t10k-images.idx3-ubyte', labels_path='data/t10k-labels.idx1-ubyte')
    savetxt(fname='data/test_images.csv', X=X, delimiter=',', fmt='%d')
    savetxt(fname='data/test_labels.csv', X=y, delimiter=',', fmt='%d')