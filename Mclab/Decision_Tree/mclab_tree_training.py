import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_mclab

X_train, X_test, X_val, y_train, y_test, y_val = import_dataset_mclab()[:6]

results = dict()
max_depth = 50
for i in range(1,max_depth):
    print('Loop {} of {}'.format(i, max_depth))
    # Fit regression model
    regressor = DecisionTreeRegressor(max_depth=i)
    regressor.fit(X_train, y_train)

    # Predict
    result = regressor.predict(X_test)

    # RMSE
    square_error = np.square(np.subtract(y_test, result)).mean()
    rmse = np.sqrt(square_error)
    results[i] = rmse


best_depth = min(results, key=results.get)
print('Best depth: {}'.format(best_depth))
regressor = DecisionTreeRegressor(max_depth=best_depth)
regressor.fit(X_train, y_train)

result = regressor.predict(X_val)
print_RMSE_MAE(y_val, result)

save_model(regressor, 'mclab_tree.sav')
save_model(results, 'mclab_tree_hiperp.sav')
