from keras import models
from sklearn.metrics import confusion_matrix
import numpy as np
from load_RD_detected_imgs import load_RD_detected_imgs
from time import time


VERBOSE = 1
# VERBOSE = 0 -> no feedback
# VERBOSE >= 1 -> print model summery and results
# VERBOSE >= 2 -> plot graphs

if VERBOSE >= 1:
    print("Reading test data...")

# Import test data
X_test, X_test_add, y_test = load_RD_detected_imgs('C:/Users/Ivan/Documents/Geolux/data/Snimka1/', 'Cube-2',
                                                   ['man', 'car', 'nothing', 'wrong_car'], [1, 0, 0, 0])

# Normalize and reshape it for ConvNet - X_train already normalized X_train.max() = 1.0 !!!
# saturation --> maybe bug, maybe ok --> data in ConvNet (0.0 - 1.0)
X_test /= X_test.max()
X_test = X_test.reshape(-1, 64, 8, 1)
X_test_add /= 256.0
X_test_add = X_test_add.reshape(-1, 2)
y_test = y_test.reshape((-1, 1))

if VERBOSE >= 1:
    print("Loading model...")

# Load saved model
model = models.load_model("models/model_trained_on_snimka1.h5")

if VERBOSE >= 1:
    model.summary()

if VERBOSE >= 1:
    print("\nClassifying test data...")

# Predict
start = time()
y_pred = np.around( model.predict([X_test, X_test_add]) )
stop = time()
correct = np.sum(y_pred == y_test)
accuracy = correct / y_test.size
con_mat = confusion_matrix(y_test, y_pred)

if VERBOSE >= 1:
    print("Number of samples: {}". format(y_test.size))
    print("Accuracy = {0:.4f}".format(accuracy))
    print("Confusion matrix:")
    print(con_mat)
    print("Execution time: {} s".format(stop-start))