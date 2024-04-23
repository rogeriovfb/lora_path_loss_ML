import matplotlib.pyplot as plt
from utils import import_dataset_mclab, print_RMSE_MAE
import joblib


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

