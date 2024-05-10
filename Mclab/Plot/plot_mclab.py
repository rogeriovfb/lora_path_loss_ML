import matplotlib.pyplot as plt
from utils import import_dataset_mclab, print_RMSE_MAE
import joblib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_IA_model (file_model, modelo):
    X_train, X_test, y_train, y_test, scaler = import_dataset_mclab()
    model = joblib.load(file_model)
    result = model.predict(X_test)
    X_test = scaler.inverse_transform(X_test)

    # Definindo a elevação como as alturas dos pontos
    alturas = X_test[:,1]

    # Definindo cores com base na elevação
    cores = alturas  # Use alturas diretamente ou normalize conforme necessário
    normalizador = plt.Normalize(vmin=min(alturas), vmax=max(alturas))
    mapa_cores = plt.cm.viridis  # Escolha um mapa de cores adequado

    # Criando a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotando os dados de saída (y_test) com a escala de cores
    scatter1 = ax.scatter(X_test[:,0], y_test, c=cores, cmap=mapa_cores, label='Dados Medidos', s=10, alpha=0.7)

    # Plotando os dados de saída (result) em uma cor fixa e mais transparente
    scatter2 = ax.scatter(X_test[:,0], result, color='red', label='Predição', s=10, alpha=0.3)

    # Adicionando barra de cores para mostrar a relação entre a cor e a elevação
    barra_cores = plt.colorbar(plt.cm.ScalarMappable(norm=normalizador, cmap=mapa_cores), ax=ax)
    barra_cores.set_label('Elevação [m]',  fontsize=16)

    # Adicionando legendas e rótulos aos eixos com tamanhos maiores de fonte
    ax.set_xlabel('Distância [km]', fontsize=16)
    ax.set_ylabel('Perda por Propagação [dB]', fontsize=16)
    ax.set_title('Predição utilizando {}'.format(modelo), fontsize=18)
    ax.legend(fontsize=14)

    # Adicionando grid
    ax.grid(True, linestyle='--', alpha=0.5)
    print_RMSE_MAE(y_test, result)


def plot_classic_models(model, titulo, log_distance: bool):
    X_train, X_test, y_train, y_test, scaler = import_dataset_mclab()
    X_test = scaler.inverse_transform(X_test)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_test[:, 0], y_test, color='blue', label='Dados Medidos', s=10, alpha=0.7)

    if log_distance:
        X_train = scaler.inverse_transform(X_train)
        result, result_oulu = model(X_train, y_train, X_test)
        plt.scatter(X_test[:, 0], result_oulu, color='green', label='Predição Oulu', s=20, alpha=1)

        print("Log Distance Fit:")
        print_RMSE_MAE(y_test, result)

        print("\nLog Distance Oulu:")
        print_RMSE_MAE(y_test, result_oulu)

    else:
        result = model(X_test)
        print_RMSE_MAE(y_test, result)

    plt.scatter(X_test[:, 0], result, color='red', label='Predição', s=20, alpha=1)

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


def plot_hiperp_3D(results_file, title='', xlabel='', ylabel=''):
    # Carregar resultados do arquivo
    results_dict = joblib.load(results_file)

    # Extrair valores de alpha, beta e rmse do dicionário de resultados
    alpha_values = []
    beta_values = []
    rmse_values = []

    for key, value in results_dict.items():
        alpha, beta, rmse = key[0], key[1], value
        alpha_values.append(alpha)
        beta_values.append(beta)
        rmse_values.append(rmse)

    # Converter listas em arrays numpy para plotagem 3D
    alpha_values = np.array(alpha_values)
    beta_values = np.array(beta_values)
    rmse_values = np.array(rmse_values)

    # Plotar gráfico 3D com o eixo beta invertido
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Criar a superfície de RMSE em função de alpha e beta
    surf = ax.plot_trisurf(alpha_values, beta_values, rmse_values, cmap='viridis', linewidth=0.2)

    # Configurar rótulos dos eixos e título
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel('RMSE [dB]', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Adicionar barra de cores para mostrar os valores de RMSE
    cbar = fig.colorbar(surf, ax=ax)
    cbar.set_label('RMSE [dB]', fontsize=14)

    # Definir os limites da escala de cores com base nos valores de RMSE
    vmin = np.min(rmse_values) - 0.5
    vmax = np.max(rmse_values) + 0.5
    surf.set_clim(vmin, vmax)

    # Encontrar a coordenada do menor RMSE
    min_rmse_idx = np.argmin(rmse_values)
    min_alpha = alpha_values[min_rmse_idx]
    min_beta = beta_values[min_rmse_idx]
    min_rmse = rmse_values[min_rmse_idx]

    # Adicionar uma estrela vermelha na coordenada do menor RMSE
    ax.scatter([min_alpha], [min_beta], [min_rmse], color='red', marker='*', s=200, label='Menor RMSE', zorder=0)

    # Mostrar as coordenadas do ponto de menor RMSE acima da estrela
    ax.text(min_alpha, min_beta, min_rmse, f'({min_alpha:.2f}, {min_beta:.2f}, {min_rmse:.2f})',
            color='black', fontsize=12, ha='center', va='top')


    # Ajustar a visualização para mudar a posição dos eixos
    ax.view_init(elev=30, azim=45)  # Exemplo de ângulos elevação e azimute

