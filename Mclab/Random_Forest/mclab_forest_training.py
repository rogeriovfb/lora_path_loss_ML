import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_mclab

X_train, X_test, y_train, y_test = import_dataset_mclab()[:4]


max_depth = 20
max_estimators = 70
results = dict()
for i in range(1, max_depth):
    print('Loop {} of {}'.format(i, max_depth))
    for k in range(1, max_estimators):
        regressor = RandomForestRegressor(n_estimators=k,
                                        max_depth=i,
                                        random_state=0)

        # fit the regressor with x and y data
        regressor.fit(X_train, y_train)

        # Predict
        result = regressor.predict(X_test)
        # RMSE
        square_error = np.square(np.subtract(y_test, result)).mean()
        rmse = np.sqrt(square_error)
        results[(i, k)] = rmse

best_hyperparams = min(results, key=results.get)
regressor = RandomForestRegressor(n_estimators=best_hyperparams[1],
                                    max_depth=best_hyperparams[0],
                                    random_state=0)

regressor.fit(X_train, y_train)
result = regressor.predict(X_test)

print_RMSE_MAE(y_test, result)
save_model(regressor, 'mclab_forest.sav')
save_model(results, 'mclab_forest_hiperp.sav')
