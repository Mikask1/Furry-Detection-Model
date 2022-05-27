from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical

plt.style.use("fivethirtyeight")

CLASS = ["furry", "non-furry"]

FOLDER_PATH = os.getcwd()+"/images/"

X = []
count = 0
for path in os.listdir(FOLDER_PATH+"furry/"):
    print(f"INFO:: Processing {count}.jpg")
    path = FOLDER_PATH+"furry/"+path

    img = Image.open(path)
    img_arr = np.array(img)

    X.append(img_arr)
    count += 1

y = np.ones((count, 1))

count = 0
for path in os.listdir(FOLDER_PATH+"non-furry/"):
    print(f"INFO:: Processing {count}.jpg")
    path = FOLDER_PATH+"non-furry/"+path

    img = Image.open(path)
    img_arr = np.array(img)

    X.append(img_arr)
    count += 1

print("INFO:: Cleaning Data..")

X = np.array(X)/255 # Normalize the pixels
y = np.vstack((y, np.zeros((count, 1))))

y = to_categorical(y)


X, x_test, y, y_test = train_test_split(
    X, y, test_size=0.1)

print("INFO:: Adding Layers..")

model = Sequential()

# First Layer Convolution Layer)
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(300, 300, 3)))

# Second Layer (Pooling Layer)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Layer (Convolution Layer)
model.add(Conv2D(32, (5, 5), activation='relu'))

# Fourth Layer (Pooling Layer)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fifth Layer (Convolution Layer)
model.add(Conv2D(64, (5, 5), activation='relu'))

# Sixth Layer (Pooling Layer)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a Flattening Layer
model.add(Flatten())

# Add a layer with 200 neurons
model.add(Dense(200, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 100 neurons
model.add(Dense(100, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 50 neurons
model.add(Dense(50, activation='relu'))

# Add a layer with 2 neurons (because we have 2 classifications)
model.add(Dense(2, activation='softmax'))

print("INFO:: Compiling..")
# Compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print("INFO:: Fitting..")
hist = model.fit(X, y, batch_size=64, epochs=10, validation_split= 0.1)

print("INFO:: Evaluating..")
model.evaluate(x_test, y_test)[1]

# Visualize accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Visualize loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()
print("INFO:: Saving..")
model.save('Furry-Detection-Model')