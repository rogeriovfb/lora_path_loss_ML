import json
import math
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


def print_RMSE_MAE(y_test2, result2):
    #Erro médio
    erro_medio = np.subtract(y_test2, result2).mean()
    print("MAE :", erro_medio)

    # Erro quadrático
    erro_quadratico = np.square(np.subtract(y_test2, result2)).mean()

    # RMSE (Raiz do Erro Quadrático Médio)
    rmse = np.sqrt(erro_quadratico)
    print("RMSE:",rmse)


dataset = pd.read_csv("G:\\Meu Drive\\Mestrado\\DISSERTAÇÃO\\DADOS\\MEDELLIN\\LoRaWAN_Path_Loss_Measurement_Campaign_No_Outliers_V2.csv")

X_col_sig= ['distance', 'sf']
X_sig = dataset[X_col_sig]
y_sig = dataset['experimental_pl']

normalizacao_sig = StandardScaler()
X_norm_sig = normalizacao_sig.fit_transform(X_sig)
X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_norm_sig, y_sig,
                                                    test_size = 0.25,
                                                    random_state = 0)

f = open('all_data_relevant.json')

# returns JSON object as
# a dictionary
data = json.load(f)
f.close()
# Iterating through the json
list_result =[]
for i in data:
    start = data[i].index(': ')+1
    end = data[i].index(', stdev')
    result = float(data[i][start + 1:end])
    if math.isnan(result):
        list_result.append(0)
    else:
        list_result.append(result)

array_result = np.array(list_result)
indices = np.argsort(array_result)[::-1][:20]

model = Sequential()
model.add(Dense(units=2, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=20000,
                kernel_initializer='normal',
                activation='relu'))

model.add(Dense(1, kernel_initializer='normal', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='Adam')
history = model.fit(X_train_sig, y_train_sig, validation_split=.1, batch_size=100, epochs=20, verbose=0)
result = model.predict(X_test_sig)

output = {}
counter = 0
for indice in indices:
    parametros = json.loads(data.get(str(indice)).split("param: ", 1)[1].replace("\'", "\""))
    # create ANN model

    model = Sequential()
    model.add(Dense(units=2, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=parametros["model__neurons"],
                    kernel_initializer='normal',
                    activation=parametros["model__activation"]))

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=parametros["model__optimizer"])
    history = model.fit(X_train_sig, y_train_sig, validation_split=.1, batch_size=100, epochs=20, verbose=0)
    result = model.predict(X_test_sig)
    result = result.reshape(232689, )

    erro_medio = np.subtract(y_test_sig, result).mean()
    erro_quadratico = np.square(np.subtract(y_test_sig, result)).mean()
    rmse = np.sqrt(erro_quadratico)
    parametros["MAE"] = erro_medio
    parametros["RMSE"] = rmse
    output[counter] = parametros
    counter +=1

with open("result_top_20_relevant.json", "w") as outfile:
    json.dump(output, outfile)
