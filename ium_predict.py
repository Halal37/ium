from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import math
import numpy as np
import os.path
import argparse
import matplotlib.pyplot as plt
import shutil

def write_list(names):
    with open('listfile.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in names))

def get_x_y(data):
    lb = LabelEncoder()

    data = data.drop(["Location 1"], axis=1)
    data = data.drop(columns=["Longitude", "Latitude", "Location", "Total Incidents", "CrimeTime", "Neighborhood", "Post", "CrimeDate", "Inside/Outside"], axis=1)
    for column_name in data.columns:
        data[column_name] = lb.fit_transform(data[column_name])
    x = data.drop('Weapon', axis=1)
    y = data['Weapon']



    return data, x, y


def predict():
    parser = argparse.ArgumentParser(description='Pred')
    parser.add_argument('-build', type=int, default=1)
    args = parser.parse_args()
    shutil.unpack_archive('baltimore.zip', 'baltimore_model', 'zip')
    model = load_model('baltimore_model')

    train = pd.read_csv('baltimore_train.csv')
    baltimore_data_test = pd.read_csv('baltimore_test.csv')
    baltimore_data_test.columns = train.columns
    baltimore_data_test, x_test, y_test = get_x_y(baltimore_data_test)
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_predicted = model.predict(x_test)
    y_predicted = np.argmax(y_predicted, axis=1)
    test_results = {}
    test_results['Weapon'] = model.evaluate(
        x_test,
        y_test, verbose=0)
    write_list(y_predicted)
    print('Accuracy : ', scores[1] * 100)
    print('Mean Absolute Error : ', metrics.mean_absolute_error(y_test, y_predicted))
    print('Root Mean Squared Error : ', math.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    if os.path.exists("metrics.csv"):
        df = pd.read_csv('metrics.csv')
        data = {
            'build': [args.build],
            'mse': metrics.mean_squared_error(y_test, y_predicted),
            'rmse': math.sqrt(metrics.mean_squared_error(y_test, y_predicted)),
            'accuracy': scores[1] * 100
        }
        row = pd.DataFrame(data)
        if df['build'].isin([int(args.build)]).any():
            df[df['build'] == args.build] = row.iloc[0]
        else:
            df = pd.concat([df, row])
            df['build'] = df['build'].astype('int')
            df.to_csv('metrics.csv', index=False)
    else:
        data = {
            'build': [args.build],
            'mse': metrics.mean_squared_error(y_test, y_predicted),
            'rmse': math.sqrt(metrics.mean_squared_error(y_test, y_predicted)),
            'accuracy': scores[1] * 100
        }
        df = pd.DataFrame(data)
        df['build'] = df['build'].astype('int')
        df.to_csv('metrics.csv', index=False)
    plt.plot(df['build'], df['mse'], label="mse")
    plt.plot(df['build'], df['rmse'], label="rmse")
    plt.plot(df['build'], df['accuracy'], label="accuracy")
    plt.legend()
    plt.show()
    plt.savefig('metrics_img.png')


predict()