import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all_data, X_test_all_data, y_train_all_data, y_test_all_data, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()[:8]

results_all_data = dict()
results_relevant = dict()
max_depth = 30

for i in range(1, max_depth):
    print('Loop {} of {}'.format(i, max_depth))
    regressor_all = DecisionTreeRegressor(max_depth=i)
    regressor_all.fit(X_train_all_data, y_train_all_data)

    regressor_relevant = DecisionTreeRegressor(max_depth=i)
    regressor_relevant.fit(X_train_relevant, y_train_relevant)

    # Predict
    result_all = regressor_all.predict(X_test_all_data)
    result_relevant = regressor_relevant.predict(X_test_relevant)

    #erro
    rmse_all = np.sqrt(np.mean(np.square(y_test_all_data - result_all)))
    rmse_relevant = np.sqrt(np.mean(np.square(y_test_relevant - result_relevant)))

    results_all_data[i] = rmse_all
    results_relevant[i] = rmse_relevant


best_depth_all_data = min(results_all_data, key=results_all_data.get)
best_depth_relevant = min(results_relevant, key=results_relevant.get)


# Fit regression model
regressor_all = DecisionTreeRegressor(max_depth=best_depth_all_data)
regressor_all.fit(X_train_all_data, y_train_all_data)

regressor_relevant = DecisionTreeRegressor(max_depth=best_depth_relevant)
regressor_relevant.fit(X_train_relevant, y_train_relevant)

# Predict
result_all = regressor_all.predict(X_test_all_data)
result_relevant = regressor_relevant.predict(X_test_relevant)

print("All Data:")
print_RMSE_MAE(y_test_all_data, result_all)
print("Relevant Factors:")
print_RMSE_MAE(y_test_relevant, result_relevant)

save_model(regressor_all, 'medellin_tree_all_data.sav')
save_model(regressor_relevant, 'medellin_tree_relevant.sav')
save_model(results_all_data, 'medellin_tree_hiperp_all_data.sav')
save_model(results_relevant, 'medellin_tree_hiperp_relevant.sav')
