import numpy as np

from utils import print_RMSE_MAE, save_model, import_dataset_mclab
from keras.models import Sequential
from keras.layers import Dense

X_train, X_test, y_train, y_test = import_dataset_mclab()[:4]


# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=3, input_dim=3, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=10000, kernel_initializer='normal', activation='relu'))

# The output neuron is a single fully connected node
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='Adamax')
history = model.fit(X_train, y_train, validation_split=.1, batch_size=100, epochs=20, verbose=0)


result = model.predict(X_test)
result = result.reshape(32765,)

print_RMSE_MAE(y_test, result)
save_model(model, 'mclab_ann_best.sav')
