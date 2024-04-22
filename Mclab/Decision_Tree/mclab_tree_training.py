import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_mclab

X_train, X_test, y_train, y_test = import_dataset_mclab()[:4]

error_list = []
max_depth = 50
for i in range (1,max_depth):
    # Fit regression model
    regressor = DecisionTreeRegressor(max_depth=i)
    regressor.fit(X_train, y_train)

    # Predict
    result = regressor.predict(X_test)

    # RMSE
    square_error = np.square(np.subtract(y_test, result)).mean()
    rmse = np.sqrt(square_error)
    error_list.append(rmse)


best_depth = error_list.index(min(error_list))+1

regressor = DecisionTreeRegressor(max_depth=best_depth)
regressor.fit(X_train, y_train)

result = regressor.predict(X_test)
print_RMSE_MAE(y_test, result)

save_model(regressor, 'mclab_tree.sav')
