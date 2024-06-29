import numpy as np

from utils import print_RMSE_MAE, save_model, import_dataset_mclab
import xgboost as xg

X_train, X_test, X_val, y_train, y_test, y_val = import_dataset_mclab()[:6]

max_depth = 30
max_estimators = 40
results = dict()

for i in range (1,max_depth):
    print('Loop {} of {}'.format(i, max_depth))
    for k in range (1,max_estimators):
        regressor = xg.XGBRegressor(n_estimators=k,
                             max_depth=i,
                             eta=0.1,
                             subsample=1,
                             colsample_bytree=1)

        # fit the regressor with x and y data
        regressor.fit(X_train, y_train)

        # Predict
        result = regressor.predict(X_test)
        # RMSE
        square_error = np.square(np.subtract(y_test, result)).mean()
        rmse = np.sqrt(square_error)
        results[(i, k)] = rmse

best_hyperparams = min(results, key=results.get)
regressor =  xg.XGBRegressor(n_estimators=best_hyperparams[1],
                             max_depth=best_hyperparams[0],
                             eta=0.1,
                             subsample=1,
                             colsample_bytree=1)
regressor.fit(X_train, y_train)

# Predict
result = regressor.predict(X_val)
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_xgboost.sav')
save_model(results, 'mclab_xgboost_hiperp.sav')
