import numpy as np

from utils import print_RMSE_MAE, save_model, import_dataset_mclab
import xgboost as xg

X_train, X_test, y_train, y_test = import_dataset_mclab()

X=[]
Y=[]
Z=[]
max_depth = 30
max_estimators = 40

for i in range (1,max_depth):
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
        X.append(i)
        Y.append(k)
        Z.append(rmse)


index_best_rmse = Z.index(min(Z))
regressor =  xg.XGBRegressor(n_estimators=Y[index_best_rmse],
                             max_depth=X[index_best_rmse],
                             eta=0.1,
                             subsample=1,
                             colsample_bytree=1)
regressor.fit(X_train, y_train)

# Predict
result = regressor.predict(X_test)

print_RMSE_MAE(y_test, result)
save_model(regressor, 'mclab_xgboost.sav')

