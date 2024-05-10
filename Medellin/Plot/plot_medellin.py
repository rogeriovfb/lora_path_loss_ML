import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from utils import import_dataset_medellin, print_RMSE_MAE
import joblib


def plot_IA_model(file_model, modelo):
    X_train_all, X_test_all, y_train_all, y_test_all, X_train_relevant, X_test_relevant, \
    y_train_relevant, y_test_relevant, scaler_all, scaler_sig = import_dataset_medellin()

    model = joblib.load(file_model)
    result = model.predict(X_test_relevant)
    X_test_relevant = scaler_sig.inverse_transform(X_test_relevant)

    sf = X_test_relevant[:, 1]

    # Definindo cores com base no SF
    cores = sf
    normalizador = plt.Normalize(vmin=min(sf), vmax=max(sf))
    mapa_cores = plt.cm.viridis  # Escolha um mapa de cores adequado

    # Criando a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotando os dados de saída (y_test) com a escala de cores
    scatter1 = ax.scatter(X_test_relevant[:, 0]/1000, y_test_relevant, marker='*',
                          c=cores, cmap=mapa_cores, label='Dados Medidos', s=50, alpha=0.7)

    # Plotando os dados de saída (result) em uma cor fixa e mais transparente
    scatter2 = ax.scatter(X_test_relevant[:, 0]/1000, result, marker='*', color='red', label='Predição', s=50, alpha=0.3)

    # Adicionando legendas e rótulos aos eixos com tamanhos maiores de fonte
    ax.set_xlabel('Distância [km]', fontsize=16)
    ax.set_ylabel('Perda por Propagação [dB]', fontsize=16)
    ax.set_title('Predição utilizando {}'.format(modelo), fontsize=18)

    # Adicionando grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Criando a legenda de cores
    unique_sf = np.unique(sf)  # Pegando valores únicos de elevação
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='{}'.format(int(elevacao)),
                              markerfacecolor=mapa_cores(normalizador(elevacao)), markersize=10)
                       for elevacao in unique_sf]
    ax.legend(handles=legend_elements, title='Fator de Espalhamento (SF)', fontsize=14, title_fontsize='large')

    print_RMSE_MAE(y_test_relevant, result)


def plot_classic_models(model, titulo, log_distance: bool):
    X_train_all, X_test_all, y_train_all, y_test_all, X_train_relevant, X_test_relevant, \
        y_train_relevant, y_test_relevant, scaler_all, scaler_sig = import_dataset_medellin()

    X_test_all = scaler_all.inverse_transform(X_test_all)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_test_all[:, 0]/1000, y_test_all, color='blue', label='Dados Medidos', s=10, alpha=0.7)

    if log_distance:
        X_train_all = scaler_all.inverse_transform(X_train_all)
        result, result_oulu = model(X_train_all, y_train_all, X_test_all)
        plt.scatter(X_test_all[:, 0]/1000, result_oulu, color='green', label='Predição Oulu', s=20, alpha=1)

        print("Log Distance Fit:")
        print_RMSE_MAE(y_test_all, result)

        print("\nLog Distance Oulu:")
        print_RMSE_MAE(y_test_all, result_oulu)

    else:
        result = model(X_test_all)
        print_RMSE_MAE(y_test_all, result)

    plt.scatter(X_test_all[:, 0]/1000, result, color='red', label='Predição', s=20, alpha=1)

    plt.title('Predição utilizando {}'.format(titulo), fontsize=18)
    plt.xlabel('Distância [km]', fontsize=16)
    plt.ylabel('Perda por Propagação [dB]', fontsize=16)
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)

def plot_hiperp_2D(results_file, title='', xlabel=''):
    results_dict = joblib.load(results_file)
    x_values = list(results_dict.keys())
    error_list = list(results_dict.values())

    # Plotar o gráfico da busca do hiperparâmetro alpha
    plt.figure(figsize=(12, 8))  # Aumentar o tamanho da figura
    plt.plot(x_values, error_list, color='blue', linestyle='-', linewidth=2)

    # Adicionar ponto com menor RMSE (marcador de estrela vermelha)
    min_rmse_index = np.argmin(error_list)
    min_rmse_alpha = x_values[min_rmse_index]
    min_rmse_value = error_list[min_rmse_index]
    plt.scatter(min_rmse_alpha, min_rmse_value, color='red', marker='*', s=200, label='Menor RMSE')

    # Adicionar texto com o valor da coordenada do ponto mínimo
    plt.text(min_rmse_alpha, min_rmse_value, f'({min_rmse_alpha:.2f}, {min_rmse_value:.2f})', fontsize=12,
             ha='center', va='top', color='black', weight='bold')

    # Estilização do gráfico
    plt.title(title, fontsize=18)  # Aumentar o tamanho do título
    plt.xlabel(xlabel, fontsize=16)  # Aumentar o tamanho do rótulo do eixo X
    plt.ylabel('RMSE [dB]', fontsize=16)  # Aumentar o tamanho do rótulo do eixo Y
    plt.xticks(fontsize=14)  # Aumentar o tamanho dos números no eixo X
    plt.yticks(fontsize=14)  # Aumentar o tamanho dos números no eixo Y
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)

