import numpy as np
from utils import import_dataset_medellin, import_dataset_medellin_raw
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

X_train_all, X_test_all, y_train_all, y_test_all, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()[:8]

dataset = import_dataset_medellin_raw()
X_col = ['distance', 'hora', "ht", "frequency", "sf", "frame_length", "temperature", "rh", "bp", "pm2_5", "toa"]

# Realizar a redução de dimensionalidade usando PCA
pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_all)
X_test_pca = pca.transform(X_test_all)

# Obter a variância explicada e a variância explicada acumulada
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plotar o gráfico de barras e a variância explicada acumulada
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})  # Tamanho da fonte

# Gráfico de barras da variância explicada por cada componente
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, label='Variância Explicada')

# Gráfico de linha da variância explicada acumulada
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='red', marker='o', linestyle='-', linewidth=2, label='Variância Explicada Acumulada')

# Adicionar anotações nos pontos da linha da variância explicada acumulada
for i, explained_var in enumerate(cumulative_variance):
    plt.annotate(f'{explained_var:.2f}', (i + 1, explained_var), textcoords="offset points",
                 xytext=(0,10), ha='center',fontsize=12)

# Configurações adicionais do gráfico
plt.xlabel('Número de Componentes Principais', fontsize=12)
plt.ylabel('Variância Explicada', fontsize=12)
plt.title('Análise de Componentes Principais', fontsize=14)
plt.xticks(range(1, len(explained_variance) + 1))
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Listas de variáveis e rótulos para os eixos X e Y
variavel = ['Temperatura', 'Umidade', 'Pressão atmosférica', 'Material particulado fino',
             'Hora', 'Altura Transmissor', 'Distância',
            'Frequência', 'Tempo no ar', 'Frame Lenght', 'Fator de Espalhamento']

eixo_X = ['Temperatura [°C]', 'Umidade [%]','Pressão [hPa]','PM 2.5 [μg/m^3]',
          'Hora', 'Altura [m]', 'Distância [m]',
          'Frequência [Hz]', 'Tempo no ar [s]', 'Frame [bytes]', 'Fator de Espalhamento']

eixo_Y = ['Sensibilidade [dB/°C]', 'Sensibilidade [dB/%]', 'Sensibilidade [dB/hPa]', 'Sensibilidade [dB/μg/m^3]',
          'Sensibilidade [dB/Hora]', 'Sensibilidade [dB/m]', 'Sensibilidade [dB/m]',
          'Sensibilidade [dB/Hz]', 'Sensibilidade [dB/s]', 'Sensibilidade [dB/bytes]', 'Sensibilidade [dB/SF]']

# Defina parâmetros de fonte e espaçamento
plt.rcParams.update({'font.size': 12})  # Tamanho da fonte
plt.rcParams.update({'axes.titlepad': 0})  # Espaçamento entre título e subplot
plt.rcParams.update({'axes.labelpad': 0})  # Espaçamento entre rótulos dos eixos e ticks

# Configuração das figuras e subplots
# Figura 1: Subplots (2x2) de Temperatura, Umidade, Pressão Atmosférica e Material Particulado Fino
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
params_fig1 = ['temperature', 'rh', 'bp', 'pm2_5']

for i, parametro in enumerate(params_fig1):
    agrupado = dataset.groupby(parametro)
    derivada = pd.DataFrame(agrupado['experimental_pl'].mean()).sort_values(by=[parametro])
    derivada["delta_pl"] = derivada.diff(axis=0).fillna(0)
    derivada["delta_param"] = derivada.index.to_series().diff().fillna(0)
    nova_col = 'sensibilidade_' + parametro
    derivada[nova_col] = (derivada["delta_pl"] / derivada["delta_param"]).fillna(0)

    X_plot = derivada.index.values
    Y_plot = derivada[nova_col].values

    axs1[i // 2, i % 2].plot(X_plot, Y_plot, 'rs', markersize=6)
    axs1[i // 2, i % 2].set_title("Sensibilidade da variável " + variavel[i])
    axs1[i // 2, i % 2].set_xlabel(eixo_X[i])
    axs1[i // 2, i % 2].set_ylabel(eixo_Y[i])
    ax = axs1.flat[i]
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Ajuste esses valores conforme necessário

# Figura 2: Subplots (1x3) de Hora, Altura Transmissor e Distância
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
params_fig2 = ['hora', 'ht', 'distance']

for i, parametro in enumerate(params_fig2):
    agrupado = dataset.groupby(parametro)
    derivada = pd.DataFrame(agrupado['experimental_pl'].mean()).sort_values(by=[parametro])
    derivada["delta_pl"] = derivada.diff(axis=0).fillna(0)
    derivada["delta_param"] = derivada.index.to_series().diff().fillna(0)
    nova_col = 'sensibilidade_' + parametro
    derivada[nova_col] = (derivada["delta_pl"] / derivada["delta_param"]).fillna(0)

    X_plot = derivada.index.values
    Y_plot = derivada[nova_col].values

    axs2[i].plot(X_plot, Y_plot, 'rs', markersize=8)
    axs2[i].set_title("Sensibilidade da variável " + variavel[i + 4])
    axs2[i].set_xlabel(eixo_X[i + 4])
    axs2[i].set_ylabel(eixo_Y[i + 4])
    ax = axs2.flat[i]
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Ajuste esses valores conforme necessário

# Figura 3: Subplots (2x2) de Frequencia, Tempo no ar, Frame Lenght e Fator de Espalhamento
fig3, axs3 = plt.subplots(2, 2, figsize=(12, 10))
params_fig3 = ['frequency', 'toa', 'frame_length', 'sf']

for i, parametro in enumerate(params_fig3):
    agrupado = dataset.groupby(parametro)
    derivada = pd.DataFrame(agrupado['experimental_pl'].mean()).sort_values(by=[parametro])
    derivada["delta_pl"] = derivada.diff(axis=0).fillna(0)
    derivada["delta_param"] = derivada.index.to_series().diff().fillna(0)
    nova_col = 'sensibilidade_' + parametro
    derivada[nova_col] = (derivada["delta_pl"] / derivada["delta_param"]).fillna(0)

    X_plot = derivada.index.values
    Y_plot = derivada[nova_col].values

    axs3[i // 2, i % 2].plot(X_plot, Y_plot, 'rs', markersize=8)
    axs3[i // 2, i % 2].set_title("Sensibilidade da variável " + variavel[i + 7])
    axs3[i // 2, i % 2].set_xlabel(eixo_X[i + 7])
    axs3[i // 2, i % 2].set_ylabel(eixo_Y[i + 7])
    ax = axs3.flat[i]
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Ajuste esses valores conforme necessário

# Exibição das figuras
#plt.tight_layout()
plt.show()
