import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


CLASS = ["furry", "non-furry"]

model = load_model('Furry-Detection-Model')

def predict(path):
    img = Image.open(path)
    img = img.resize((300, 300))
    img = img.convert("RGB")
    img_arr = np.array(img)

    x = np.array([img_arr])/255
    pred = model.predict(x)

    max_index = np.argmax(pred[0])

    return CLASS[max_index]

def predict_folder(path):

    x = []
    for img_path in os.listdir(path):
        img = Image.open(path+"/"+img_path)
        img_arr = np.array(img)
        x.append(img_arr)
    
    x = np.array(x)/255

    pred = model.predict(x)
    res = []
    for i in pred:
        max_index = np.argmax(i)

        res.append(CLASS[max_index])
    
    return res

def train(furry, non):
    furry_path, furry_code = furry
    non_path, non_code = non

    x = []
    count = 0
    for img_path in os.listdir(furry_path):
        img = Image.open(furry_path+"/"+img_path)
        img_arr = np.array(img)
        x.append(img_arr)
        count += 1

    y = np.ones((count, 1))

    count = 0
    for path in os.listdir(non_path):
        img = Image.open(non_path+"/"+path)
        img_arr = np.array(img)
        x.append(img_arr)
        count += 1
    
    x = np.array(x)/255
    y = np.vstack((y, np.zeros((count, 1))))

    y = to_categorical(y)

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
    hist = model.fit(x, y, batch_size=64, epochs=10, validation_split=0.2)

    print("INFO:: Evaluating..")
    for i in model.evaluate(x_test, y_test):
        print(i)

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
    model.save()

train(("jes", 1), ("non", 1))




