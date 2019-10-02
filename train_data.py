import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from sklearn.metrics import confusion_matrix
from load_RD_detected_imgs import load_RD_detected_imgs


VERBOSE = 1
# VERBOSE = 0 -> no feedback
# VERBOSE >= 1 -> print model summery and results
# VERBOSE >= 2 -> plot graphs


############################################### Preprocessing ##################################################
if VERBOSE >= 1:
    print("Reading training data...")

# Read images
X_train, X_train_add, y_train = load_RD_detected_imgs('C:/Users/Ivan/Documents/Geolux/data/Snimka1/', 'Cube-0',
                                                      ['man', 'car', 'nothing', 'wrong_car'], [1, 0, 0, 0])

# X_train2, X_train_add2, y_train2 = load_RD_detected_imgs('C:/Users/Ivan/Documents/Geolux/data/Snimka2/', 'Cube-0',
#                                                          ['man', 'car', 'nothing', 'wrong_car'], [1, 0, 0, 0])
# X_train = np.append(X_train, X_train2)
# X_train_add = np.append(X_train_add, X_train_add2)
# y_train = np.append(y_train, y_train2)

# Normalize data and reshape it for ConvNet - X_train already normalized X_train.max() = 1.0 !!!
# X_train /= X_train.max()
X_train = X_train.reshape(-1, 64, 8, 1)
X_train_add /= 256.0
X_train_add = X_train_add.reshape(-1, 2)


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
img = layers.Input(shape = (64, 8, 1))
conv1 = layers.Conv2D(32, (4,2), activation = 'relu') (img)
maxpool1 = layers.MaxPooling2D(pool_size = (3,2)) (conv1)
dropout1 = layers.Dropout(rate = 0.5) (maxpool1)
conv2 = layers.Conv2D(64, (4,2), activation = 'relu') (dropout1)
maxpool2 = layers.MaxPool2D(pool_size = (2,1)) (conv2)
dropout2 = layers.Dropout(rate = 0.5) (maxpool2)
conv3 = layers.Conv2D(64, (4,2), activation = 'relu') (dropout2)
flatten = layers.Flatten() (conv3)
dense1 = layers.Dense(units = 128, activation = 'relu') (flatten)
dropout3 = layers.Dropout(rate = 0.5) (dense1)

additional_args = layers.Input(shape = (2,) )
concat = layers.Concatenate()([dropout3, additional_args])

dense2 = layers.Dense(units = 1, activation = 'sigmoid') (concat)

model = models.Model(inputs = [img, additional_args], outputs = [dense2])

if VERBOSE >= 1:
    model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])



############################################### Train the CNN ##############################################
if VERBOSE == 0:
    verb = 0
else:
    verb = 2
history = model.fit([X_train, X_train_add], y_train, epochs = 100, batch_size = 128, class_weight={0: 1., 1: 1.4},
                    validation_data = ([X_val, X_val_add], y_val), verbose = verb)

################################################ Analisys ##################################################

if VERBOSE >= 1:
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    y_pred = np.around( model.predict([X_val, X_val_add]) )
    con_mat = confusion_matrix(y_val, y_pred)   # , labels=['1', '0']
    print("Confusion matrix:")
    print(con_mat)

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

# model.save('model_trained_on_snimka1_and_snimka2_Cube_0.h5')
exit(0)

if VERBOSE >= 1:
    print("Reading test data...")

# Import test data
X_test, X_test_add, y_test = load_RD_detected_imgs('C:/Users/Ivan/Documents/Geolux/data/Snimka2/', 'Cube-0',
                                                   ['man', 'car', 'nothing', 'wrong_car'], [1, 0, 0, 0])

# Normalize data and reshape it for ConvNet !!! X_test already normalized (saturated)
X_test /= X_test.max()
X_test = X_test.reshape(-1, 64, 8, 1)
X_test_add /= 256.0
X_test_add = X_test_add.reshape(-1, 2)
y_test = y_test.reshape((-1, 1))

if VERBOSE >= 1:
    print("\nClassifying test data...")

# Predict
y_pred = np.around( model.predict([X_test, X_test_add]) )
correct = np.sum(y_pred == y_test)
accuracy = correct / y_test.size
print("Number of samples: {}". format(y_test.size))
print("Accuracy = {0:.4f}".format(accuracy))

con_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(con_mat / y_test.size)