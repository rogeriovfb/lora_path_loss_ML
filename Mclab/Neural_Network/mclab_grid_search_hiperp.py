import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import json

def create_model(neurons, activation, optimizer):
    # create model
    model = Sequential()
    model.add(Dense(units=3, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=neurons, kernel_initializer='normal', activation=activation))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def print_RMSE_MAE(y_test, result):
    #Erro médio
    erro_medio = np.subtract(y_test, result).mean()
    print("MAE :", erro_medio)

    # Erro quadrático
    erro_quadratico = np.square(np.subtract(y_test, result)).mean()

    # RMSE (Raiz do Erro Quadrático Médio)
    rmse = np.sqrt(erro_quadratico)
    print("RMSE:",rmse)

def plot_pred_med(y_test, result, titulo):
    plt.figure()
    X_plot = normalizacao.inverse_transform(X_test)
    plt.plot(X_plot[:,0], y_test, 'g*', label='Medição')
    plt.plot(X_plot[:,0], result, 'b*', label='Predição')
    plt.title(titulo)
    plt.xlabel('Distancia [km]')
    plt.ylabel('Path Loss [dBm]')
    plt.legend(loc='best',fancybox=True, shadow=True)


dataset = pd.read_csv("G:\\Meu Drive\\Mestrado\\DISSERTAÇÃO\\DADOS\\MCLAB\\dataset.csv")

X_col = ['distance', 'elevations', 'antenna_height']
X = dataset[X_col]
y = dataset['path_loss']
X = X.values
y = y.values

normalizacao = StandardScaler()
X_norm = normalizacao.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y,
                                                    test_size=0.25,
                                                    random_state=0)

model = KerasRegressor(model=create_model, epochs=20, batch_size=50, verbose=1)

neurons = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(model__neurons=neurons, model__activation=activation, model__optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train,  validation_split=.1)

print("Melhores parâmetros encontrados: ", grid_result.best_params_)
print("Melhor pontuação de MSE: ", grid_result.best_score_)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

all_data = {}
counter = 0
for mean, stdev, param in zip(means, stds, params):
    texto = "Mean: {}, stdev: {},  param: {}".format(mean, stdev, param)
    all_data[counter] = texto
    counter += 1
    print(texto)

best_model = grid_result.best_estimator_
mse_test = best_model.score(X_test, y_test)

print("MSE: ", grid_result.best_score_)

result = best_model.predict(X_test)
print(result)

result = result.reshape(32765,)

print_RMSE_MAE(y_test, result)
plot_pred_med(y_test, result, "Rede Neural")

with open("best_model.json", "w") as outfile:
    json.dump(grid_result.best_params_, outfile)

with open("all_data.json", "w") as outfile:
    json.dump(all_data, outfile)
