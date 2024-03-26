import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile

def print_RMSE_MAE(y_test, result):
    mae = np.mean(np.abs(y_test - result))
    print("MAE :", mae)
    rmse = np.sqrt(np.mean(np.square(y_test - result)))
    print("RMSE:",rmse)


def plot_pred_med(X_test, y_test, result, titulo): #TODO
    plt.figure()
    X_plot = X_test
    plt.plot(X_plot[:,0], y_test, 'g*', label='Medição')
    plt.plot(X_plot[:,0], result, 'b*', label='Predição')

    plt.title(titulo)
    plt.xlabel('Distancia [km]')
    plt.ylabel('Path Loss [dBm]')
    plt.legend(loc='best',fancybox=True, shadow=True)


def save_model (model, filename):
    joblib.dump(model, filename)


def import_dataset_mclab():
    zf = zipfile.ZipFile('..\\..\\Data\\dataset_mclab.zip')
    dataset = pd.read_csv(zf.open('dataset_mclab.csv'))


    X_col = ['distance', 'elevations', 'antenna_height']
    X = dataset[X_col]
    y = dataset['path_loss']
    X = X.values
    y = y.values

    scaler = StandardScaler()
    X_scaler= scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y,
                                                        test_size=0.25,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def import_dataset_medellin():
    zf = zipfile.ZipFile('..\\..\\Data\\dataset_medellin.zip')
    dataset = pd.read_csv(zf.open('dataset_medellin.csv'))

    dataset['hora'] = pd.to_datetime(dataset['timestamp'], format='%d/%m/%Y %H:%M').dt.hour

    X_col = ['distance', 'hora', "ht", "frequency", "sf", "frame_length", "temperature", "rh", "bp", "pm2_5", "toa"]
    X = dataset[X_col]
    y = dataset['experimental_pl']

    normalizacao_todos = StandardScaler()
    X_norm = normalizacao_todos.fit_transform(X)
    X_train_todos, X_test_todos, y_train_todos, y_test_todos = train_test_split(X_norm, y,
                                                                                test_size=0.25,
                                                                                random_state=0)

    X_col_sig = ['distance', 'sf']
    X_sig = dataset[X_col_sig]
    y_sig = dataset['experimental_pl']

    normalizacao_sig = StandardScaler()
    X_norm_sig = normalizacao_sig.fit_transform(X_sig)
    X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_norm_sig, y_sig,
                                                                        test_size=0.25,
                                                                        random_state=0)
    return X_train_todos, X_test_todos, y_train_todos, y_test_todos, X_train_sig, X_test_sig, y_train_sig, y_test_sig