import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile


def print_RMSE_MAE(y_test, result):
    result = result.reshape(len(result),)
    mae = np.mean(np.abs(y_test - result))
    print("MAE: {:.2f} dB".format(mae))
    rmse = np.sqrt(np.mean(np.square(y_test - result)))
    print("RMSE: {:.2f} dB".format(rmse))


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
    #X_train, X_test, y_train, y_test = train_test_split(X_scaler, y,
    #                                                    test_size=0.25,
    #                                                    random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.20, random_state=1)
    # 0.125 x 0.8 = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

    save_model(scaler, 'scaler_mclab.sav')
    return X_train, X_test,X_val, y_train, y_test, y_val, scaler


def import_dataset_medellin_raw():
    zf = zipfile.ZipFile('..\\..\\Data\\dataset_medellin.zip')
    dataset = pd.read_csv(zf.open('dataset_medellin.csv'))

    dataset['hora'] = pd.to_datetime(dataset['timestamp'], format='%d/%m/%Y %H:%M').dt.hour
    return dataset


def import_dataset_medellin():
    dataset = import_dataset_medellin_raw()

    X_col = ['distance', 'hora', "ht", "frequency", "sf", "frame_length", "temperature", "rh", "bp", "pm2_5", "toa"]
    X = dataset[X_col]
    y = dataset['experimental_pl']

    normalizacao_todos = StandardScaler()
    X_norm = normalizacao_todos.fit_transform(X)
    #X_train_todos, X_test_todos, y_train_todos, y_test_todos = train_test_split(X_norm, y,
    #                                                                            test_size=0.25,
    #                                                                            random_state=0)

    X_train_todos, X_test_todos, y_train_todos, y_test_todos = train_test_split(X_norm, y, test_size=0.20, random_state=1)
    # 0.125 x 0.8 = 0.1
    X_train_todos, X_val_todos, y_train_todos, y_val_todos = train_test_split(X_train_todos, y_train_todos, test_size=0.125, random_state=1)

    X_col_sig = ['distance', 'sf']
    X_sig = dataset[X_col_sig]
    y_sig = dataset['experimental_pl']

    normalizacao_sig = StandardScaler()
    X_norm_sig = normalizacao_sig.fit_transform(X_sig)
    #X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_norm_sig, y_sig,
    #                                                                    test_size=0.25,
    #                                                                    random_state=0)

    X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_norm_sig, y_sig, test_size=0.20, random_state=1)

    X_train_sig, X_val_sig, y_train_sig, y_val_sig = train_test_split(X_train_sig, y_train_sig, test_size=0.125, random_state=1)


    return X_train_todos, X_test_todos, X_val_todos, y_train_todos, y_test_todos, y_val_todos, \
           X_train_sig, X_test_sig, X_val_sig, y_train_sig, y_test_sig, y_val_sig, \
           normalizacao_todos, normalizacao_sig

def plot_medellin_dataframe():
    dataset = import_dataset_medellin_raw()

    # Criar uma figura para os três primeiros gráficos
    fig1, axs1 = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))

    # Colunas, títulos e rótulos para os três gráficos
    X_col_1 = ['distance', 'ht', 'experimental_pl']
    titles_1 = ['Distância', 'Altura Transmissor', 'Perda por Propagação']
    x_labels_1 = ['Distância [km]', 'Altura [m]', 'Perda por Propagação [dB]']

    # Iterar sobre cada coluna e plotar o histograma usando Seaborn
    for i, col in enumerate(X_col_1):
        sns.histplot(dataset[col].dropna(), color='blue', ax=axs1[i])
        sns.histplot(dataset[col].dropna(), kde=True, color='black', ax=axs1[i])
        axs1[i].set_ylabel('Amostras', fontsize=14)
        axs1[i].set_title(f'Distribuição de {titles_1[i]}', fontsize=16)
        axs1[i].set_xlabel(x_labels_1[i], fontsize=14)

    # Ajustar o espaçamento
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # Criar uma figura para os gráficos restantes
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(18, 21))

    # Colunas, títulos e rótulos para os gráficos restantes
    X_col_2 = ['hora', 'frequency', 'sf', 'frame_length', 'temperature', 'rh', 'bp', 'pm2_5', 'toa']
    titles_2 = ['Hora', 'Frequência', 'SF', 'Frame Length', 'Temperatura', 'Umidade Relativa', 'Pressão atm.', 'PM2.5',
                'Time on Air']
    x_labels_2 = ['Hora', 'Frequência [Hz]', 'SF', 'Frame Length [bytes]', 'Temperatura [°C]', 'Umidade [%]',
                  'Pressão [hPa]', 'PM2.5 [μg/m^3]', 'TOA [s]']

    # Iterar sobre cada coluna e plotar o histograma usando Seaborn
    for i, col in enumerate(X_col_2):
        row = i // 3
        col_index = i % 3
        sns.histplot(dataset[col].dropna(), color='blue', ax=axs2[row, col_index])
        sns.histplot(dataset[col].dropna(), kde=True, color='black', ax=axs2[row, col_index])
        axs2[row, col_index].set_ylabel('Amostras', fontsize=14)
        axs2[row, col_index].set_title(f'Distribuição de {titles_2[i]}', fontsize=16)
        axs2[row, col_index].set_xlabel(x_labels_2[i], fontsize=14)

    # Ajustar o espaçamento
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)


def plot_mclab_dataframe():
    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    zf = zipfile.ZipFile('..\\..\\Data\\dataset_mclab.zip')
    df = pd.read_csv(zf.open('dataset_mclab.csv'))

    # --------------------------------------------------------
    # Global style configuration (consistent with other figures)
    # --------------------------------------------------------
    sns.set(style="whitegrid")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13
    })

    # --------------------------------------------------------
    # Create subplots
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --------------------------------------------------------
    # Distance distribution
    # --------------------------------------------------------
    sns.histplot(df['distance'], color='dodgerblue', ax=axes[0, 0])
    sns.histplot(df['distance'], kde=True, color='black', ax=axes[0, 0])

    axes[0, 0].set_title('Distance Distribution')
    axes[0, 0].set_xlabel('Distance [km]')
    axes[0, 0].set_ylabel('Samples')

    # --------------------------------------------------------
    # Elevation distribution
    # --------------------------------------------------------
    sns.histplot(df['elevations'], color='dodgerblue', ax=axes[0, 1])
    sns.histplot(df['elevations'], kde=True, color='black', ax=axes[0, 1])

    axes[0, 1].set_title('Elevation Distribution')
    axes[0, 1].set_xlabel('Elevation [m]')
    axes[0, 1].set_ylabel('Samples')

    # --------------------------------------------------------
    # Antenna height distribution
    # --------------------------------------------------------
    sns.histplot(df['antenna_height'], color='dodgerblue', ax=axes[1, 0])
    sns.histplot(df['antenna_height'], kde=True, color='black', ax=axes[1, 0])

    axes[1, 0].set_title('Antenna Height Distribution')
    axes[1, 0].set_xlabel('Antenna Height [m]')
    axes[1, 0].set_ylabel('Samples')

    # --------------------------------------------------------
    # Path loss distribution
    # --------------------------------------------------------
    sns.histplot(df['path_loss'], color='dodgerblue', ax=axes[1, 1])
    sns.histplot(df['path_loss'], kde=True, color='black', ax=axes[1, 1])

    axes[1, 1].set_title('Path Loss Distribution')
    axes[1, 1].set_xlabel('Path Loss [dB]')
    axes[1, 1].set_ylabel('Samples')

    # --------------------------------------------------------
    # Layout and save
    # --------------------------------------------------------
    plt.tight_layout()

    plt.savefig(
        'fig1_dataset_feature_distributions.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
