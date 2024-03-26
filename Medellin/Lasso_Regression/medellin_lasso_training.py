import numpy as np
from sklearn.linear_model import Lasso
from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all, X_test_all, y_train_all, y_test_all, \
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = import_dataset_medellin()

error_list_all = []
error_list_relevant = []
max_alpha = 50

for i in np.arange(0, 5.0, 0.01):
    # Fit regression model
    regressor_all = Lasso(alpha=i)
    regressor_relevant = Lasso(alpha=i)

    regressor_all.fit(X_train_all, y_train_all)
    regressor_relevant.fit(X_train_relevant, y_train_relevant)

    # Predict
    result_all = regressor_all.predict(X_test_all)
    result_relevant = regressor_relevant.predict(X_test_relevant)

    # RMSE
    square_error_all = np.square(np.subtract(y_test_all, result_all)).mean()
    square_error_relevant = np.square(np.subtract(y_test_relevant, result_relevant)).mean()
    rmse_all = np.sqrt(square_error_all)
    rmse_relevant = np.sqrt(square_error_relevant)
    error_list_all.append(rmse_all)
    error_list_relevant.append(rmse_relevant)


# Fit regression model
regressor_all = Lasso(alpha=0.01)
regressor_all.fit(X_train_all, y_train_all)

regressor_relevant = Lasso(alpha=0.01)
regressor_relevant.fit(X_train_relevant, y_train_relevant)

# Predict
result_all = regressor_all.predict(X_test_all)
print_RMSE_MAE(y_test_all, result_all)
save_model(regressor_all, 'medellin_lasso_all_data.sav')

result_relevant = regressor_relevant.predict(X_test_relevant)
print_RMSE_MAE(y_test_relevant, result_relevant)
save_model(regressor_relevant, 'medellin_lasso_relevant.sav')

