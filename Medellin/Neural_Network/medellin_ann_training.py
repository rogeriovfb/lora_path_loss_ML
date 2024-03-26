from utils import print_RMSE_MAE, save_model, import_dataset_medellin
from keras.models import Sequential
from keras.layers import Dense

X_train_all_data, X_test_all_data, y_train_all_data, y_test_all_data, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()



# Fit regression model
# All Data
model_all_data = Sequential()
model_all_data.add(Dense(units=11, input_dim=11, kernel_initializer='normal', activation='relu'))
model_all_data.add(Dense(units=100,
                    kernel_initializer='normal',
                    activation='relu'))

model_all_data.add(Dense(1, kernel_initializer='normal', activation='linear'))
model_all_data.compile(loss='mean_squared_error', optimizer='Adam')
history = model_all_data.fit(X_train_all_data, y_train_all_data, validation_split=.1, batch_size=100, epochs=20, verbose=0)

# Predict
result_all_data = model_all_data.predict(X_test_all_data)
result_all_data = result_all_data.reshape(232689, )

# Relevant
model_relevant = Sequential()
model_relevant.add(Dense(units=2, input_dim=2, kernel_initializer='normal', activation='relu'))
model_relevant.add(Dense(units=10,
                    kernel_initializer='normal',
                    activation='sigmoid'))

model_relevant.add(Dense(1, kernel_initializer='normal', activation='linear'))
model_relevant.compile(loss='mean_squared_error', optimizer='Nadam')
history = model_relevant.fit(X_train_relevant, y_train_relevant, validation_split=.1, batch_size=100, epochs=20, verbose=0)

# Predict
result_relevant = model_relevant.predict(X_test_relevant)
result_relevant = result_relevant.reshape(232689, )


print("All Data:")
print_RMSE_MAE(y_test_all_data, result_all_data)
print("Relevant Factors:")
print_RMSE_MAE(y_test_relevant, result_relevant)

save_model(model_all_data, 'medellin_ann_all_data.sav')
save_model(model_relevant, 'medellin_ann_relevant.sav')
