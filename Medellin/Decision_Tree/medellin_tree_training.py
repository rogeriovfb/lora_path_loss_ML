import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all, X_test_all, y_train_all, y_test_all, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()


error_list_all = []
error_list_relevant = []

max_depth = 30
for i in range(1, max_depth):
    # Fit regression model
    regressor_all = DecisionTreeRegressor(max_depth=i)
    regressor_all.fit(X_train_all, y_train_all)

    regressor_relevant = DecisionTreeRegressor(max_depth=i)
    regressor_relevant.fit(X_train_relevant, y_train_relevant)

    # Predict
    result_all = regressor_all.predict(X_test_all)
    result_relevant = regressor_relevant.predict(X_test_relevant)

    #erro
    rmse_all = np.sqrt(np.mean(np.square(y_test_all - result_all)))
    rmse_relevant = np.sqrt(np.mean(np.square(y_test_relevant - result_relevant)))

    error_list_all.append(rmse_all)
    error_list_relevant.append(rmse_relevant)

best_depth_all = error_list_all.index(min(error_list_all))+1
best_depth_relevant = error_list_relevant.index(min(error_list_relevant))+1

# Fit regression model
regressor_all = DecisionTreeRegressor(max_depth=best_depth_all)
regressor_all.fit(X_train_all, y_train_all)

regressor_relevant = DecisionTreeRegressor(max_depth=best_depth_relevant)
regressor_relevant.fit(X_train_relevant, y_train_relevant)

# Predict
result_all = regressor_all.predict(X_test_all)
result_relevant = regressor_relevant.predict(X_test_relevant)

print("All Data:")
print_RMSE_MAE(y_test_all, result_all)
print("Relevant Factors:")
print_RMSE_MAE(y_test_relevant, result_relevant)

save_model(regressor_all, 'medellin_tree_all_data.sav')
save_model(regressor_relevant, 'medellin_tree_relevant.sav')
