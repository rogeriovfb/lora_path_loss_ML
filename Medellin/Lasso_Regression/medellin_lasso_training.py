import numpy as np
from sklearn.linear_model import Lasso
from utils import print_RMSE_MAE, save_model, import_dataset_medellin

X_train_all_data, X_test_all_data, X_val_all_data, y_train_all_data, y_test_all_data, y_val_all_data,\
X_train_relevant, X_test_relevant, X_val_relevant, y_train_relevant, y_test_relevant, y_val_relevant = \
    import_dataset_medellin()[:12]

results_all = dict()
results_relevant = dict()
max_alpha = 5

for i in np.arange(0.01, max_alpha, 0.01):
    print('Alpha {} of {}'.format(i, max_alpha))
    # Fit regression model
    regressor_all = Lasso(alpha=i)
    regressor_relevant = Lasso(alpha=i)

    regressor_all.fit(X_train_all_data, y_train_all_data)
    regressor_relevant.fit(X_train_relevant, y_train_relevant)

    # Predict
    result_all = regressor_all.predict(X_test_all_data)
    result_relevant = regressor_relevant.predict(X_test_relevant)

    # RMSE
    square_error_all = np.square(np.subtract(y_test_all_data, result_all)).mean()
    square_error_relevant = np.square(np.subtract(y_test_relevant, result_relevant)).mean()
    rmse_all = np.sqrt(square_error_all)
    rmse_relevant = np.sqrt(square_error_relevant)

    results_all[i] = rmse_all
    results_relevant[i] = rmse_relevant


# Fit regression model
regressor_all = Lasso(alpha=0.01)
regressor_all.fit(X_train_all_data, y_train_all_data)

regressor_relevant = Lasso(alpha=0.01)
regressor_relevant.fit(X_train_relevant, y_train_relevant)

# Predict
result_all = regressor_all.predict(X_val_all_data)
print_RMSE_MAE(y_val_all_data, result_all)
save_model(regressor_all, 'medellin_lasso_all_data.sav')
save_model(results_all, 'medellin_lasso_hiperp_all_data.sav')

result_relevant = regressor_relevant.predict(X_val_relevant)
print_RMSE_MAE(y_val_relevant, result_relevant)
save_model(regressor_relevant, 'medellin_lasso_relevant.sav')
save_model(results_relevant, 'medellin_lasso_hiperp_relevant.sav')
