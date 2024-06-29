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

    # Colunas que você quer plotar como histogramas
    X_col = ['distance', 'hora', 'ht', 'frequency', 'sf', 'frame_length', 'temperature', 'rh', 'bp', 'pm2_5', 'toa']

    # Listas para títulos e rótulos do eixo x
    titles = ['Distância', 'Hora', 'Altura Transmissor', 'Frequência', 'SF', 'Frame Length', 'Temperatura', 'Umidade Relativa', 'Pressão atm.', 'PM2.5', 'Time on Air']
    x_labels = ['Distância [km]', 'Hora', 'Altura [m]', 'Frequência [Hz]', 'SF', 'Frame Length [bytes]', 'Temperatura [°C]', 'Umidade[%]', 'Pressão [hPa]', 'PM2.5 [μg/m^3]',
                'TOA [s]']

    # Configurações padrão de estilo Seaborn
    sns.set(style='whitegrid')

    # Ajustando o tamanho das fontes
    plt.rcParams.update({'font.size': 14})  # Definindo o tamanho da fonte para 14 pontos

    # Calcular o número de linhas e colunas para os subplots
    num_cols = 3  # Defina o número de colunas desejado
    num_rows = (len(X_col) + num_cols - 1) // num_cols  # Calcula o número de linhas necessário

    # Criar uma figura e os subplots usando Seaborn
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 7 * num_rows))

    # Iterar sobre cada coluna e plotar o histograma usando Seaborn
    for i, col in enumerate(X_col):
        row = i // num_cols
        col_index = i % num_cols
        sns.histplot(dataset[col].dropna(), color='blue', ax=axs[row, col_index])
        sns.histplot(dataset[col].dropna(), kde=True, color='black', ax=axs[row, col_index])
        axs[row, col_index].set_ylabel('Amostras', fontsize=14)
        axs[row, col_index].set_title(f'Histograma de {titles[i]}', fontsize=16)  # Título com fonte maior
        axs[row, col_index].set_xlabel(x_labels[i], fontsize=14)  # Rótulo do eixo x com fonte maior

    # Remover subplots não utilizados, se houver
    if len(X_col) < num_rows * num_cols:
        for i in range(len(X_col), num_rows * num_cols):
            fig.delaxes(axs[i // num_cols, i % num_cols])

    # Ajustar o espaçamento entre subplots
    plt.subplots_adjust(left=0.24, bottom=0.064, right=0.75, top=0.95, wspace=0.41, hspace=0.53)  # Ajusta os espaçamentos superior e inferior entre os subplots


def plot_mclab_dataframe():
    # Carregando o DataFrame a partir do arquivo zip
    zf = zipfile.ZipFile('..\\..\\Data\\dataset_mclab.zip')
    df = pd.read_csv(zf.open('dataset_mclab.csv'))

    # Configurando o estilo dos gráficos utilizando o seaborn
    sns.set(style="whitegrid")

    # Ajustando o tamanho das fontes
    plt.rcParams.update({'font.size': 14})  # Definindo o tamanho da fonte para 14 pontos

    # Criando subplots para os gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histograma para 'distance' com barras azuis
    sns.histplot(df['distance'], color='blue', ax=axes[0, 0])
    axes[0, 0].set_title('Distribuição de Distância', fontsize=16)
    axes[0, 0].set_xlabel('Distância [km]', fontsize=14)
    axes[0, 0].set_ylabel('Amostras', fontsize=14)

    # Gráfico de densidade (kde) para 'distance' com linha preta
    sns.histplot(df['distance'], kde=True, color='black', ax=axes[0, 0])

    # Histograma para 'elevations' com barras azuis
    sns.histplot(df['elevations'], color='blue', ax=axes[0, 1])
    axes[0, 1].set_title('Distribuição de Elevações', fontsize=16)
    axes[0, 1].set_xlabel('Elevações [m]', fontsize=14)
    axes[0, 1].set_ylabel('Amostras', fontsize=14)

    # Gráfico de densidade (kde) para 'elevations' com linha preta
    sns.histplot(df['elevations'], kde=True, color='black', ax=axes[0, 1])

    # Histograma para 'antenna_height' com barras azuis
    sns.histplot(df['antenna_height'], color='blue', ax=axes[1, 0])
    axes[1, 0].set_title('Distribuição de Altura da Antena', fontsize=16)
    axes[1, 0].set_xlabel('Altura da Antena [m]', fontsize=14)
    axes[1, 0].set_ylabel('Amostras', fontsize=14)

    # Gráfico de densidade (kde) para 'antenna_height' com linha preta
    sns.histplot(df['antenna_height'], kde=True, color='black', ax=axes[1, 0])

    # Histograma para 'path_loss' com barras azuis
    sns.histplot(df['path_loss'], color='blue', ax=axes[1, 1])
    axes[1, 1].set_title('Distribuição de Perda por propagação', fontsize=16)
    axes[1, 1].set_xlabel('Perda por propagação [dB]', fontsize=14)
    axes[1, 1].set_ylabel('Amostras', fontsize=14)

    # Gráfico de densidade (kde) para 'path_loss' com linha preta
    sns.histplot(df['path_loss'], kde=True, color='black', ax=axes[1, 1])

    # Aumentando o contraste das linhas nos gráficos de densidade
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Definindo a largura das linhas para 1.5
            spine.set_edgecolor('black')  # Definindo a cor das linhas como preto

    # Ajustando o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()