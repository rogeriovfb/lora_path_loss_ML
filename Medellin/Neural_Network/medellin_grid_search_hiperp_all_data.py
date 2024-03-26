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
    model.add(Dense(units=11, input_dim=11, kernel_initializer='normal', activation='relu'))
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

def plot_pred_med(X_test, y_test, result, titulo):
  plt.figure()
  #X_plot = normalizacao.inverse_transform(X_test)
  X_plot = X_test
  plt.plot(X_plot[:,0], y_test, 'g*', label='Medição')
  plt.plot(X_plot[:,0], result, 'b*', label='Predição')

  plt.title(titulo)
  plt.xlabel('Distancia [km]')
  plt.ylabel('Path Loss [dBm]')
  plt.legend(loc='best',fancybox=True, shadow=True)

dataset = pd.read_csv("G:\\Meu Drive\\Mestrado\\DISSERTAÇÃO\\DADOS\\MEDELLIN\\LoRaWAN_Path_Loss_Measurement_Campaign_No_Outliers_V2.csv")

#dataset = dataset.sample(10000)

dataset['hora'] = pd.to_datetime(dataset['timestamp'], format='%d/%m/%Y %H:%M').dt.hour

X_col= ['distance', 'hora', "ht", "frequency", "sf","frame_length", "temperature", "rh", "bp", "pm2_5",  "toa"]
X = dataset[X_col]
y = dataset['experimental_pl']

normalizacao_todos= StandardScaler()
X_norm = normalizacao_todos.fit_transform(X)
X_train_todos, X_test_todos, y_train_todos, y_test_todos = train_test_split(X_norm, y,
                                                    test_size = 0.25,
                                                    random_state = 0)

X_col_sig= ['distance', 'sf']
X_sig = dataset[X_col_sig]
y_sig = dataset['experimental_pl']

normalizacao_sig = StandardScaler()
X_norm_sig = normalizacao_sig.fit_transform(X_sig)
X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_norm_sig, y_sig,
                                                    test_size = 0.25,
                                                    random_state = 0)

model = KerasRegressor(model=create_model, epochs=20, batch_size=50, verbose=1)

neurons = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

#neurons = [10, 20]
#activation = ['softmax', 'softplus']
#optimizer = ['Adam', 'Adamax', 'Nadam']

param_grid = dict(model__neurons=neurons, model__activation=activation, model__optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result_todos = grid.fit(X_train_todos, y_train_todos,  validation_split=.1)


print("Melhores parâmetros encontrados com todas entradas: ", grid_result_todos.best_params_)
print("Melhor pontuação de MSE com todas entradas: ", grid_result_todos.best_score_)

means_todos = grid_result_todos.cv_results_['mean_test_score']
stds_todos = grid_result_todos.cv_results_['std_test_score']
params_todos = grid_result_todos.cv_results_['params']

all_data_todos = {}
counter = 0
for mean, stdev, param in zip(means_todos, stds_todos, params_todos):
    texto = "Mean: {}, stdev: {},  param: {}".format(mean, stdev, param)
    all_data_todos[counter] = texto
    counter += 1
    print(texto)

best_model_todos = grid_result_todos.best_estimator_
mse_test_todos = best_model_todos.score(X_test_todos, y_test_todos)

print("MSE: ", grid_result_todos.best_score_)

result_todos = best_model_todos.predict(X_test_todos)
print(result_todos)

result_todos = result_todos.reshape(232689,)

print_RMSE_MAE(y_test_todos, result_todos)

with open("best_model_all_data.json", "w") as outfile:
    json.dump(grid_result_todos.best_params_, outfile)

with open("all_data_all.json", "w") as outfile:
    json.dump(all_data_todos, outfile)

