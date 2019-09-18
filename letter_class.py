import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#% matplotlib inline

from sklearn.model_selection import  train_test_split
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

VERBOSE = 2
# VERBOSE = 0 -> no feedback
# VERBOSE >= 1 -> print model summery and results
# VERBOSE >= 2 -> plot graphs

if VERBOSE >= 1:
    print("Reading training data...")

# Read input
X_train = np.genfromtxt("data/train_images.csv", delimiter=',')
y_train = np.genfromtxt("data/train_labels.csv", delimiter=',')

# Normalize data and reshape it for ConvNet
X_train /= 255.0
X_train = X_train.reshape(-1, 28, 28, 1)

if VERBOSE >= 2:
    plt.figure()
    plt.hist(y_train, bins=10, color='blue')
    plt.xlabel("digits")
    plt.ylabel("number of reppetitions of a digit")
    plt.title("Training data set")
    plt.show()

# Make labels categorical variables
y_train = to_categorical(y_train, num_classes = 10)

# Divide training set into train and validation set (10% in validation set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)

# Build a CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1), data_format="channels_last"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))

if VERBOSE >= 1:
    model.summary()

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

# Train the CNN
if VERBOSE == 0:
    verb = 0
else:
    verb = 2
history = model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_val, y_val), verbose = verb)

if VERBOSE >= 1:
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

if VERBOSE >= 2:
    epochs = range(1, 21)

    plt.figure()
    plt.plot(epochs, loss, 'ko', label='Training Loss')
    plt.plot(epochs, val_loss, 'k', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, acc, 'yo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()


# Import test data
X_test = np.genfromtxt("data/test_images.csv", delimiter=',')
y_test = np.genfromtxt("data/test_labels.csv", delimiter=',')

# Normalize and reshape data
X_test = X_test/255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Predict
results = model.predict(X_test)

# One hot vector to digit number
results = np.argmax(results, axis = 1)

# Get accuracy
correct = np.sum(results == y_test)
accuracy = correct / y_test.size

print("Accuracy = %.4f".format(accuracy))







