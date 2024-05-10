import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all_data, X_test_all_data, y_train_all_data, y_test_all_data, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()[:8]

results_all_data = dict()
results_relevant = dict()

max_depth = 25
max_estimators = 25

for i in range(1, max_depth):
    print('Loop {} of {}'.format(i, max_depth))
    for k in range(1, max_estimators):
        regressor_all_data = RandomForestRegressor(n_estimators=k,
                                    max_depth=i,
                                    random_state=0)

        regressor_relevant = RandomForestRegressor(n_estimators=k,
                                    max_depth=i,
                                    random_state=0)

        # fit the regressor with x and y data
        regressor_all_data.fit(X_train_all_data, y_train_all_data)
        regressor_relevant.fit(X_train_relevant, y_train_relevant)

        # Predict
        result_all_data = regressor_all_data.predict(X_test_all_data)
        result_relevant = regressor_relevant.predict(X_test_relevant)

        #erro
        square_error_all_data = np.square(np.subtract(y_test_all_data, result_all_data)).mean()
        rmse_all_data = np.sqrt(square_error_all_data)

        square_error_relevant = np.square(np.subtract(y_test_relevant, result_relevant)).mean()
        rmse_relevant = np.sqrt(square_error_relevant)

        results_all_data[(i, k)] = rmse_all_data
        results_relevant[(i, k)] = rmse_relevant

best_hyperparams_all_data = min(results_all_data, key=results_all_data.get)
best_hyperparams_relevant = min(results_relevant, key=results_relevant.get)

# Fit regression model
regressor_all_data = RandomForestRegressor(n_estimators=best_hyperparams_all_data[1],
                                            max_depth=best_hyperparams_all_data[0],
                                            random_state=0)
regressor_all_data.fit(X_train_all_data, y_train_all_data)

# Predict
result_all_data = regressor_all_data.predict(X_test_all_data)

regressor_relevant = RandomForestRegressor(n_estimators=best_hyperparams_relevant[1],
                                            max_depth=best_hyperparams_relevant[0],
                                            random_state=0)
regressor_relevant.fit(X_train_relevant, y_train_relevant)

# Predict
result_relevant = regressor_relevant.predict(X_test_relevant)

print("All Data:")
print_RMSE_MAE(y_test_all_data, result_all_data)
print("Relevant Factors:")
print_RMSE_MAE(y_test_relevant, result_relevant)

save_model(regressor_all_data, 'medellin_forest_all_data.sav')
save_model(regressor_relevant, 'medellin_forest_relevant.sav')
save_model(results_all_data, 'medellin_forest_hiperp_all_data.sav')
save_model(results_relevant, 'medellin_forest_hiperp_relevant.sav')