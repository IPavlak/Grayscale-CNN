import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.model_selection import  train_test_split
from keras import models
from keras import layers

VERBOSE = 2
# VERBOSE = 0 -> no feedback
# VERBOSE >= 1 -> print model summery and results
# VERBOSE >= 2 -> plot graphs

if VERBOSE >= 1:
    print("Reading training data...")

############################################### Preprocessing ##################################################
# Read images
X_train = np.genfromtxt("data/train_images.csv", delimiter=',')
y_train = np.genfromtxt("data/train_labels.csv", delimiter=',')

# Read additional arguments
X_train_add = np.genfromtxt("data/train_images.csv", delimiter=',')

# Normalize data and reshape it for ConvNet
X_train /= 255.0
X_train = X_train.reshape(-1, 64, 8, 1)

if VERBOSE >= 2:
    plt.figure()
    plt.hist(y_train, bins=2, color='blue')
    plt.xlabel("digits")
    plt.ylabel("number of repetitions of a digit")
    plt.title("Training data set")
    plt.show()

# Divide training set into train and validation set (10% in validation set) and shuffle the data
X_train, X_val, y_train, y_val, X_train_add, X_val_add = \
    train_test_split(X_train, y_train, X_train_add, test_size = 0.1, random_state = 2)


########################################### Build a CNN model #################################################
img = layers.Input(shape = (28, 28, 1))
conv1 = layers.Conv2D(32, (3,3), activation = 'relu') (img)
maxpool1 = layers.MaxPooling2D(pool_size = (2,2)) (conv1)
dropout1 = layers.Dropout(rate = 0.5) (maxpool1)
conv2 = layers.Conv2D(64, (3,3), activation = 'relu') (dropout1)
maxpool2 = layers.MaxPool2D(pool_size = (2,2)) (conv2)
dropout2 = layers.Dropout(rate = 0.5) (maxpool2)
conv3 = layers.Conv2D(64, (3,3), activation = 'relu') (dropout2)
flatten = layers.Flatten() (conv3)
dense1 = layers.Dense(units = 128, activation = 'relu') (flatten)
dropout3 = layers.Dropout(rate = 0.5) (dense1)

additional_args = layers.Input(shape = (2,) )
concat = layers.Concatenate()([dropout3, additional_args])

dense2 = layers.Dense(units = 1, activation = 'softmax') (concat)

model = models.Model(inputs = [img, additional_args], outputs = [dense2])

if VERBOSE >= 1:
    model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])



############################################### Train the CNN ##############################################
if VERBOSE == 0:
    verb = 0
else:
    verb = 2
history = model.fit([X_train, X_train_add], y_train, epochs = 20, batch_size = 128,
                    validation_data = ([X_val, X_val_add], y_val), verbose = verb)

################################################ Analisys ##################################################

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







