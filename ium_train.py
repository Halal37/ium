# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import shutil


def get_x_y(data):

    lb = LabelEncoder()
    data = data.drop(["Location 1"], axis=1)
    data = data.drop(columns=["Longitude", "Latitude", "Location", "Total Incidents", "CrimeTime", "Neighborhood", "Post", "CrimeDate", "Inside/Outside"], axis=1)
    for column_name in data.columns:
        data[column_name] = lb.fit_transform(data[column_name])
    x = data.drop('Weapon', axis=1)
    y = data['Weapon']



    return data, x, y


def train_model():

    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-validation_split', type=float, default=0.2)
    args = parser.parse_args()
    
    train = pd.read_csv('baltimore_train.csv')

    data_train, x_train, y_train = get_x_y(train)
    normalizer = tf.keras.layers.Normalization(axis=1)
    normalizer.adapt(np.array(x_train))
    model = Sequential(normalizer)
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation="softmax"))
    model.compile(Adam(learning_rate=args.lr), loss='sparse_categorical_crossentropy', metrics = ['accuracy'] )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        validation_split=args.validation_split)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    model.save('baltimore_model')
    shutil.make_archive('baltimore', 'zip', 'baltimore_model')


train_model()

